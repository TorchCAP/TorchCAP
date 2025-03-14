# The code is adapted from torch/_inductor/comm_analysis.py
# and modified to work with torch.fx

import torch
import torch.fx as fx
import functools
import math
from enum import IntEnum

from torch._inductor.utils import get_gpu_type
from torch.distributed.tensor._dtensor_spec import DTensorSpec

from torchcap.fx_utils import size_of
from torchcap.cluster_env import MeshTopology

c10d_ops = torch.ops.c10d
funcol_native = torch.ops._c10d_functional


class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2


class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2


@functools.lru_cache
def get_gpu_type() -> NVIDIA_GPU_TYPE:
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run) or ""
    if "V100" in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif "A100" in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif "H100" in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    else:
        # for other gpu types, assume Ampere
        return NVIDIA_GPU_TYPE.AMPERE


collective_ops = {
    c10d_ops._allgather_base_,
    c10d_ops._reduce_scatter_base_,
    c10d_ops.allgather_,
    c10d_ops.allgather_coalesced_,
    c10d_ops.allgather_into_tensor_coalesced_,
    c10d_ops.allreduce_,
    c10d_ops.allreduce_coalesced_,
    c10d_ops.alltoall_,
    c10d_ops.alltoall_base_,
    c10d_ops.broadcast_,
    c10d_ops.gather_,
    c10d_ops.scatter_,
    c10d_ops.reduce_,
    c10d_ops.reduce_scatter_,
    c10d_ops.reduce_scatter_tensor_coalesced_,
    funcol_native.all_gather_into_tensor,
    funcol_native.all_reduce,
    funcol_native.reduce_scatter_tensor,
    funcol_native.all_to_all_single,
    funcol_native.broadcast,
    funcol_native.wait_tensor,
}


def get_collective_type(node: fx.Node) -> NCCL_COLL:
    # if not isinstance(node, ir._CollectiveKernel):
    #     raise ValueError(f"node is not a collective kernel: {node}")

    kernel_name = node.target.__name__
    assert kernel_name is not None
    if "all_reduce" in kernel_name:
        return NCCL_COLL.ALL_REDUCE
    elif "all_gather" in kernel_name:
        return NCCL_COLL.ALL_GATHER
    elif "reduce_scatter" in kernel_name:
        return NCCL_COLL.REDUCE_SCATTER
    else:
        raise ValueError(f"Unsupported collective kernel: {kernel_name}")


def get_collective_input_size_bytes(node: fx.Node) -> int:
    sz_bytes = 0
    for arg in node.args:
        if isinstance(arg, fx.Node):
            sz_bytes += size_of(arg)
    return sz_bytes


def get_collective_group_size(node: fx.Node) -> int:
    from torch.distributed.distributed_c10d import _get_group_size_by_name
    return _get_group_size_by_name(node.constant_args[-1])


def is_collective(node: fx.Node) -> bool:
    return node.target._overloadpacket in collective_ops


def is_wait(node: fx.Node) -> bool:
    return node.target._overloadpacket == funcol_native.wait_tensor


####################################################################################################################
# The following code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
####################################################################################################################


class NCCL_HW(IntEnum):
    NVLINK = 0
    PCI = 1
    NET = 2


class NCCL_ALGO(IntEnum):
    TREE = 0
    RING = 1


class NCCL_PROTO(IntEnum):
    # The ordering and enum values here matches original in
    # https://github.com/NVIDIA/nccl/blob/0b083e52096c387bad7a5c5c65b26a9dca54de8c/src/include/devcomm.h#L28
    # For difference between these protocols, see https://github.com/NVIDIA/nccl/issues/281#issuecomment-571816990
    LL = 0  # Low-latency
    # LL128 = 1   # Low-latency 128-byte
    # SIMPLE = 2


# Latencies in us
# len(NCCL_ALGO) x len(NCCL_PROTO)
# NOTE: use array instead of tensor to prevent incompatibility with fake mode
baseLat = [
    # Tree
    [
        6.8,  # LL
    ],
    # Ring
    [
        6.6,  # LL
    ],
]

# Latencies in us
# len(NCCL_HW) x len(NCCL_ALGO) x len(NCCL_PROTO)
hwLat = [
    # NVLINK
    [
        [0.6],  # Tree (LL)
        [0.6],  # Ring (LL)
    ],
    # PCI
    [
        [1.0],  # Tree (LL)
        [1.0],  # Ring (LL)
    ],
    # NET
    [
        [5.0],  # Tree (LL)
        [2.7],  # Ring (LL)
    ],
]


# LL128 max BW per channel
llMaxBws = [
    # Volta-N1/Intel-N2/Intel-N4
    [
        39.0,
        39.0,
        20.4,
    ],
    # Ampere-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
    # Hopper-N1/AMD-N2/AMD-N4
    [
        87.7,
        22.5,  # avg of ring & tree
        19.0,
    ],
]


def estimate_nccl_collective_runtime(node: fx.Node) -> float:
    """
    Returns estimated NCCL collective runtime in nanoseconds (ns).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
    """
    tensor_storage_size_bytes = get_collective_input_size_bytes(node)
    # Convert bytes to GB
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024

    # Currently assumes each node has 8 gpus. And when >1 node is used, assumes each node uses all 8 gpus.
    # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    num_gpus_per_node = 8
    group_size = get_collective_group_size(node)
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return 0

    # Assumes ring algorithm
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL
    coll = get_collective_type(node)

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns

    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw

    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    # NOTE: each step of ring algorithm is synchronized,
    # and is bottlenecked by the slowest link which is the inter-node interconnect.
    # hence when nNodes >= 2, bw is inter-node bandwidth.
    # NOTE: the original code in https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc
    # have this as `if nNodes <= 2` which seems wrong. Corrected it here.
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2  # Assume # channels is 2
    busBw = nChannels * bw

    # Various model refinements
    busBw = min(
        llMaxBw,
        busBw
        * (1.0 / 4.0 if (nNodes > 1 or coll == NCCL_COLL.ALL_REDUCE) else 1.0 / 3.0),
    )

    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    ratio = (1.0 * nRanks) / nsteps  # type: ignore[possibly-undefined]
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============
    intraHw = NCCL_HW.NVLINK

    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nInterSteps = nNodes - 1

    # First compute latency in us; then at the end, convert it to ns
    latency = baseLat[nccl_algo][nccl_proto]
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto]
    interLat = hwLat[NCCL_HW.NET][nccl_algo][nccl_proto]

    # Inter-node rings still have to launch nsteps * net overhead.
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat  # type: ignore[possibly-undefined]
    # Convert us to ns
    latency_ns = latency * 1e3

    # =============== final result ===============
    transport_ns = tensor_storage_size_GB / bandwidth_GB_per_ns
    return transport_ns + latency_ns


def all_reduce_cost(op_bytes: int, mesh_topo: MeshTopology, mesh_dim: int) -> float:
    assert mesh_topo is not None and mesh_dim in mesh_topo.comm_model
    num_devices = mesh_topo.mesh_shape[mesh_dim]
    bytes_gb = op_bytes / 2**30 # size in GB
    x = 2 * (num_devices - 1) / num_devices * bytes_gb
    return mesh_topo.comm_model[mesh_dim](x)


def all_gather_cost(op_bytes: int, mesh_topo: MeshTopology, mesh_dim: int) -> float:
    assert mesh_topo is not None and mesh_dim in mesh_topo.comm_model
    num_devices = mesh_topo.mesh_shape[mesh_dim]
    bytes_gb = op_bytes / 2**30 # size in GB
    x = (num_devices - 1) / num_devices * bytes_gb
    return mesh_topo.comm_model[mesh_dim](x)


def reduce_scatter_cost(op_bytes: int, mesh_topo: MeshTopology, mesh_dim: int) -> float:
    assert mesh_topo is not None and mesh_dim in mesh_topo.comm_model
    num_devices = mesh_topo.mesh_shape[mesh_dim]
    bytes_gb = op_bytes / 2**30 # size in GB
    x = (num_devices - 1) / num_devices * bytes_gb
    return mesh_topo.comm_model[mesh_dim](x)


def all_to_all_cost(op_bytes: int, mesh_topo: MeshTopology, mesh_dim: int) -> float:
    assert mesh_topo is not None and mesh_dim in mesh_topo.comm_model
    num_devices = mesh_topo.mesh_shape[mesh_dim]
    bytes_gb = op_bytes / 2**30 # size in GB
    x = (num_devices - 1) / num_devices * bytes_gb
    return mesh_topo.comm_model[mesh_dim](x)


def broadcast_cost(op_bytes: int, mesh_topo: MeshTopology, mesh_dim: int) -> float:
    assert mesh_topo is not None and mesh_dim in mesh_topo.comm_model
    bytes_gb = op_bytes / 2**30 # size in GB
    x = bytes_gb
    return mesh_topo.comm_model[mesh_dim](x)


def reshard_cost(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    mesh_topo: MeshTopology,
) -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trival (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
    if src_spec.mesh != dst_spec.mesh:
        # make infinite cost if meshes are not same
        # TODO: see if we want to support this once there's cross mesh communication
        return float("inf")

    if src_spec.is_replicated():
        # short-cut:
        # comm cost is 0 if current spec is already full replication
        return 0.0

    def spec_to_bytes(spec: DTensorSpec) -> int:
        assert spec.tensor_meta is not None, "spec should have tensor meta defined!"
        return spec.tensor_meta.dtype.itemsize * math.prod(spec.shape)

    cost = 0.0
    comm_bytes_gb = (
        spec_to_bytes(src_spec) / src_spec.num_shards / 1024 / 1024 / 1024
    )
    # Transformation that considered for redistribute cost:
    # 1. allgather 2. alltoall
    # 3. allreduce 4. reduce_scatter
    for i, (current, target) in enumerate(
        zip(src_spec.placements, dst_spec.placements)
    ):
        if current == target:
            continue

        num_devices_on_mesh_dim = mesh_topo.mesh_shape[i]
        if current.is_shard() and target.is_replicate():
            # allgather gives larger comm bytes
            comm_bytes_gb *= num_devices_on_mesh_dim
            # add up allgather comm cost
            cost += all_gather_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_shard() and target.is_shard():
            # should be alltoall comm, since we haven't implement it yet, add penalty
            # to favor allgather instead
            cost += all_to_all_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_partial() and target.is_replicate():
            # add up allreduce comm cost
            cost += all_reduce_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_partial() and target.is_shard():
            # add up reduce_scatter comm cost
            cost += reduce_scatter_cost(comm_bytes_gb, mesh_topo, i)
            # after reduce_scatter the comm bytes for further collectives halved.
            comm_bytes_gb /= num_devices_on_mesh_dim
        elif current.is_shard() and target.is_partial():
            # ban shard -> partial as it does not make sense to perform
            # this redistribute
            return float("inf")

    return cost