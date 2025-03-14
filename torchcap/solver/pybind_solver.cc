#include <vector>
#include <set>
#include <string>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ortools/base/logging.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/util/sorted_interval_list.h"


namespace py = pybind11;


const bool DEBUG = false;


template <typename K, typename V>
const V& FindOrDie(const std::unordered_map<K, V>& map, const K& key) {
  auto it = map.find(key);
  CHECK(it != map.end()) << "Missing key " << key; // Crash ok
  return it->second;
}


std::string CpStatusString(int32_t status) {
  switch (status) {
    case 0: return "UNKNOWN";
    case 1: return "MODEL_INVALID";
    case 2: return "FEASIBLE";
    case 3: return "INFEASIBLE";
    case 4: return "OPTIMAL";
    default: return "OTHER";
  }
}


namespace operations_research {
namespace sat {


std::vector<IntVar> MakeIntVars(
  operations_research::sat::CpModelBuilder &model,
  int64_t num_vars,
  int64_t lb,
  int64_t ub,
  std::string name
) {
  std::vector<IntVar> vars;
  for (int i = 0; i < num_vars; ++i) {
    vars.push_back(
      model.NewIntVar(Domain(lb, ub)).WithName(absl::StrCat(name, "_", i)));
  }
  return vars;
}

std::vector<BoolVar> MakeBoolVars(
  operations_research::sat::CpModelBuilder &model,
  int64_t num_vars,
  std::string name
) {
  std::vector<BoolVar> vars;
  for (int i = 0; i < num_vars; ++i) {
    vars.push_back(
      model.NewBoolVar().WithName(absl::StrCat(name, "_", i)));
  }
  return vars;
}


struct CpSolverResult {
  std::unordered_map<std::string, std::tuple<bool, bool>> param_policy;
  std::unordered_set<int> modules_to_wrap;
  std::unordered_set<int> modules_to_recompute;
  int peak_device_memory;
  double objective;
};


IntVar bool_int_prod(
  CpModelBuilder &model, BoolVar x, IntVar y
) {
  auto z = model.NewIntVar(y.Domain());
  model.AddEquality(z, y).OnlyEnforceIf(x);
  model.AddEquality(z, 0).OnlyEnforceIf(x.Not());
  return z;
}


std::optional<CpSolverResult> CallCpSolver(
  const std::vector<int64_t> &F,
  const std::vector<int64_t> &B,
  const std::vector<int64_t> &P,
  const std::vector<int64_t> &TA,
  const std::vector<int64_t> &TMP,
  const std::vector<int64_t> &SA,
  const std::vector<int64_t> &I,
  const std::vector<int64_t> &O,
  int MAX_P,
  int DP_SIZE,
  int D2D_BW,
  int H2D_BW,
  int M_GPU,
  int M_CPU,
  const std::unordered_map<std::string, int> &named_parameters,
  const std::vector<std::vector<std::string>> &module_to_param,
  const std::vector<std::vector<int>> &ad_matrx,
  const std::vector<int> &pre2pos,
  const std::unordered_set<std::string> &params_not_to_shard
) {
  int BYTES_PER_OPT = 12 / 4;
  int EXTRA_CPU_BUF = 1.5;

  int N = F.size();

  CpModelBuilder model;
  model.SetName("cp_model");

  // Sets a time limit of 10 seconds.
  SatParameters parameters;
  parameters.set_max_time_in_seconds(300.0);
  parameters.set_log_search_progress(DEBUG);

  // Upper bounds
  int H2D_UB = int(P[0] / H2D_BW);
  int D2D_UB = int(P[0] / D2D_BW) * 100;

  int MAX_TA = *std::max_element(TA.begin(), TA.end());
  int MAX_TMP = *std::max_element(TMP.begin(), TMP.end());
  int MAX_A = MAX_TA + MAX_TMP;

  std::vector<int> pos2pre(pre2pos.size());
  for (int i = 0; i < pre2pos.size(); i++) {
    int j = pre2pos[i];
    pos2pre[j] = i;
  }

  std::cout << absl::StrFormat(
    "N=%d, F[0]=%d, B[0]=%d, P[0]=%d, MAX_P=%d, MAX_TA=%d, MAX_TMP=%d, M_GPU=%d, M_CPU=%d, DP_SIZE=%d, H2D_BW=%d, D2D_BW=%d, H2D_UB=%d, D2D_UB=%d",
    N, F[0], B[0], P[0], MAX_P, MAX_TA, MAX_TMP, M_GPU, M_CPU, DP_SIZE, H2D_BW, D2D_BW, H2D_UB, D2D_UB) << std::endl;

  // Decision variables

  // x[n]: indicator if parameter n is sharded
  std::unordered_map<std::string, BoolVar> x(named_parameters.size());
  for (auto &it: named_parameters) {
    x[it.first] = model.NewBoolVar().WithName(absl::StrCat("x_", it.first));
  }

  // y[n]: indicator if parameter n is offloaded
  std::unordered_map<std::string, BoolVar> y(named_parameters.size());
  for (auto &it: named_parameters) {
    y[it.first] = model.NewBoolVar().WithName(absl::StrCat("y_", it.first));
  }

  // wrap[i]: indicator if module i is wrapped
  std::vector<BoolVar> wrap = MakeBoolVars(model, N, "wrap");

  // ckpt[i]: indicator if module i is checkpointed
  std::vector<BoolVar> ckpt = MakeBoolVars(model, N, "ckpt");

  // p_sha[i]: sharded parameter size during module i
  std::vector<IntVar> p_sha = MakeIntVars(model, N, 0, P[0], "p_sha");

  // p_off[i]: offloaded parameter size during module i
  std::vector<IntVar> p_off = MakeIntVars(model, N, 0, P[0], "p_off");

  // p_rep[i]: replicated parameter size during module i
  std::vector<IntVar> p_rep = MakeIntVars(model, N, 0, P[0], "p_rep");

  // a: total activation memory during training
  std::vector<IntVar> a = MakeIntVars(model, N, 0, MAX_A, "a");

  // m_gpu: peak device memory during training
  IntVar m_gpu = model.NewIntVar(Domain(0, M_GPU*10)).WithName("m_gpu");

  // m_cpu: peak host memory during training
  IntVar m_cpu = model.NewIntVar(Domain(0, M_CPU*10)).WithName("m_cpu");

  // ag[i]: all-gather communication time during module i
  std::vector<IntVar> ag = MakeIntVars(model, N, 0, D2D_UB, "ag");

  // rs[i]: Reduce-scatter communication time during module i
  std::vector<IntVar> rs = MakeIntVars(model, N, 0, D2D_UB, "rs");

  // rd[i]: Reload communication time during module i
  std::vector<IntVar> rd = MakeIntVars(model, N, 0, H2D_UB, "rd");

  // fw_ag[i]: forward prefetch all-gather communication time during module i
  std::vector<IntVar> fw_ag = MakeIntVars(model, N, 0, D2D_UB, "fw_ag");

  // bw_ag[i]: backward prefetch all-gather communication time during module i
  std::vector<IntVar> bw_ag = MakeIntVars(model, N, 0, D2D_UB, "bw_ag");

  // bw_rs[i]: previous module's reduce-scatter communication time during module i
  std::vector<IntVar> bw_rs = MakeIntVars(model, N, 0, D2D_UB, "bw_rs");

  // fw_rd[i]: forward preload communication time during module i
  std::vector<IntVar> fw_rd = MakeIntVars(model, N, 0, H2D_UB, "fw_rd");

  // bw_rd[i]: backward preload communication time during module i
  std::vector<IntVar> bw_rd = MakeIntVars(model, N, 0, H2D_UB, "bw_rd");

  // c_fw[i]: communication overhead during forward pass
  std::vector<IntVar> c_fw = MakeIntVars(model, N, 0, D2D_UB, "c_fw");

  // c_bw[i]: communication overhead during backward pass
  std::vector<IntVar> c_bw = MakeIntVars(model, N, 0, D2D_UB, "c_bw");

  // Constraints

  // [Constraint] Only replicated parameters can be offloaded
  for (auto &[n, p]: named_parameters) {
    model.AddImplication(FindOrDie(y, n), FindOrDie(x, n).Not());
  }

  // [Constraint] Zero-sized parameters are ignored
  for (auto &[n, p]: named_parameters) {
    if (p == 0) {
      model.AddEquality(FindOrDie(x, n), 0);
      model.AddEquality(FindOrDie(y, n), 0);
    }
  }

  // [Constraint] User specified parameters that cannot be sharded
  for (auto &[n, p]: named_parameters) {
    if (params_not_to_shard.find(n) != params_not_to_shard.end()) {
      model.AddEquality(FindOrDie(x, n), 0);
    }
  }

  // [Constraint] Sharded parameter size of module i
  IntVar total_sharded = model.NewIntVar(Domain(0, P[0])).WithName("total_sharded");
  {
    std::vector<BoolVar> vars;
    std::vector<int64_t> coeffs;
    for (auto &n: module_to_param[0]) {
      vars.push_back(FindOrDie(x, n));
      coeffs.push_back(FindOrDie(named_parameters, n));
    }
    model.AddEquality(total_sharded, LinearExpr::WeightedSum(vars, coeffs));
  }

  // [Constrian] Offloaded parameter size of module i
  IntVar total_offloaded = model.NewIntVar(Domain(0, P[0])).WithName("total_offloaded");
  {
    std::vector<BoolVar> vars;
    std::vector<int64_t> coeffs;
    for (auto &n: module_to_param[0]) {
      vars.push_back(FindOrDie(y, n));
      coeffs.push_back(FindOrDie(named_parameters, n));
    }
    model.AddEquality(total_offloaded, LinearExpr::WeightedSum(vars, coeffs));
  }

  // [Constraint] Replicated parameter size during module i
  IntVar total_replicated = model.NewIntVar(Domain(0, P[0])).WithName("total_replicated");
  model.AddEquality(total_replicated, P[0] - total_sharded - total_offloaded);

  // [Constraint] Total optimizer state size during training
  IntVar total_sharded_per_dp = model.NewIntVar(Domain(0, P[0])).WithName("total_sharded_per_dp");
  model.AddDivisionEquality(total_sharded_per_dp, total_sharded, DP_SIZE);

  // [Constraint] No nested checkpointing
  model.AddEquality(ckpt[0], 0);
  for (int i = 1; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      if (ad_matrx[i][j] == 1) {
        model.AddLessOrEqual(ckpt[i] + ckpt[j], 1);
      }
    }
  }

  // [Constraint] Do not checkpoint leaf modules
  for (int i = 1; i < N; i++) {
    bool is_leaf = true;
    for (int j = 0; j < N; j++) {
      if (i != j && ad_matrx[i][j] == 1) {
        is_leaf = false;
        break;
      }
    }
    if (is_leaf) {
      model.AddEquality(ckpt[i], 0);
    }
  }

  // [Constraint] Total activation memory during module i
  for (int i = 0; i < N; i++) {
    std::vector<BoolVar> vars;
    std::vector<int64_t> coeffs;
    for (int j = 0; j < N; j++) {
      if (pre2pos[j] < pre2pos[i]) {
        // expr += ckpt[j] * (SA[j] - I[j] - O[j]);
        vars.push_back(ckpt[j]);
        coeffs.push_back(SA[j] - I[j] - O[j]);
      }
    }
    model.AddEquality(a[i], TA[i] + TMP[i] - LinearExpr::WeightedSum(vars, coeffs));
  }

  // [Constraint] Maximum activation memory
  IntVar max_a = model.NewIntVar(Domain(0, MAX_A)).WithName("max_a");
  model.AddMaxEquality(max_a, a);

  // [Constraint] Recomputation time during training
  IntVar rcp = model.NewIntVar(Domain(0, F[0])).WithName("rcp");
  model.AddEquality(rcp, LinearExpr::WeightedSum(ckpt, F));

  // [Constraint] Peak device memory during training
  // `BYTES_PER_OPT+2` to account for the gradients and `MAX_P*2` to account for the prefetched parameters
  model.AddEquality(
    m_gpu,
    (total_sharded_per_dp + total_replicated) * (BYTES_PER_OPT + 2) + MAX_P*2 + max_a
  );

  // [Constraint] Peak host memory during training
  model.AddEquality(
    m_cpu,
    total_offloaded * int(BYTES_PER_OPT * EXTRA_CPU_BUF)
  );

  // Host memory capacity constraint
  model.AddLessOrEqual(m_cpu, M_CPU);

  // Stage 1 objective
  IntVar max_m = model.NewIntVar(Domain(0, M_GPU*10)).WithName("max_m");
  model.AddMaxEquality(max_m, {m_gpu, M_GPU});
  model.Minimize(max_m);

  // Solve stage 1
  CpSolverResponse resp;
  resp = SolveWithParameters(model.Build(), parameters);

  LOG(INFO) << absl::StrFormat("Stage 1 completed (status: %s, objective: %f, wall-time (s): %f)", 
    CpStatusString(resp.status()), resp.objective_value(), resp.wall_time());

  // Return if infeasible stage 1
  if (resp.status() != CpSolverStatus::FEASIBLE && resp.status() != CpSolverStatus::OPTIMAL) {
    return std::nullopt;
  }

  // Return if the solution is not a memory feasible solution
  if (SolutionIntegerValue(resp, max_m) > M_GPU || SolutionIntegerValue(resp, m_cpu) > M_CPU) {
    LOG(INFO) << absl::StrFormat("No memory feasible solution found (max_m: %d, M_GPU: %d, m_cpu: %d, M_CPU: %d)", 
      SolutionIntegerValue(resp, max_m), M_GPU, SolutionIntegerValue(resp, m_cpu), M_CPU);
    if (DEBUG) {
      std::cout << absl::StrFormat("total_sharded_per_dp=%d, total_replicated=%d, MAX_P=%d, max_a=%d",
        SolutionIntegerValue(resp, total_sharded_per_dp),
        SolutionIntegerValue(resp, total_replicated),
        MAX_P,
        SolutionIntegerValue(resp, max_a)
      ) << std::endl;
    }
    return std::nullopt;
  }

  for (auto &[n, p]: named_parameters) {
    model.AddHint(x[n], SolutionBooleanValue(resp, x[n]));
    model.AddHint(y[n], SolutionBooleanValue(resp, y[n]));
  }
  for (int i = 0; i < N; i++) {
    model.AddHint(ckpt[i], SolutionBooleanValue(resp, ckpt[i]));
  }

  // parameters.set_fix_variables_to_their_hinted_value(true);
  model.ClearObjective();

  // Device memory capacity constraint
  model.AddLessOrEqual(m_gpu, M_GPU);

  // [Constraint] Root module is always a wrapped unit
  model.AddEquality(wrap[0], 1);

  // [Constraint] No nested wrapped units (except the root module)
  for (int i = 1; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      if (ad_matrx[i][j] == 1) {
        model.AddLessOrEqual(wrap[i] + wrap[j], 1);
      }
    }
  }

  // [Constraint] A module can be wrapped if and only if it has offloaded or sharded parameters
  for (int i = 0; i < N; i++) {
    std::vector<BoolVar> xy_i;
    for (auto &n: module_to_param[i]) {
      xy_i.push_back(FindOrDie(x, n));
      xy_i.push_back(FindOrDie(y, n));
    }
    model.AddLessOrEqual(wrap[i], LinearExpr::Sum(xy_i));
  }

  // [Constraint] Sharded parameter size of each wrapped unit
  for (int i = 1; i < N; i++) {
    LinearExpr rhs;
    for (auto &n: module_to_param[i]) {
      rhs += FindOrDie(x, n) * FindOrDie(named_parameters, n);
    }
    model.AddEquality(p_sha[i], rhs).OnlyEnforceIf(wrap[i]);
    model.AddEquality(p_sha[i], 0).OnlyEnforceIf(wrap[i].Not());
  }
  // p_sha[0] == sum(x[n] * p[n]) - p_sha[1:]
  LinearExpr p_sha_0;
  for (auto &[n, p]: named_parameters) {
    p_sha_0 += FindOrDie(x, n) * p;
  }
  model.AddEquality(p_sha[0], p_sha_0 - LinearExpr::Sum(
    absl::MakeConstSpan(p_sha).subspan(1, N-1)
  ));

  // [Constraint] Offloaded parameter size of each wrapped unit
  for (int i = 1; i < N; i++) {
    LinearExpr rhs;
    for (auto &n: module_to_param[i]) {
      rhs += FindOrDie(y, n) * FindOrDie(named_parameters, n);
    }
    model.AddEquality(p_off[i], rhs).OnlyEnforceIf(wrap[i]);
    model.AddEquality(p_off[i], 0).OnlyEnforceIf(wrap[i].Not());
  }
  // p_off[0] == sum(y[n] * p[n]) - p_off[1:]
  LinearExpr p_off_0;
  for (auto &[n, p]: named_parameters) {
    p_off_0 += FindOrDie(y, n) * p;
  }
  model.AddEquality(p_off[0], p_off_0 - LinearExpr::Sum(
    absl::MakeConstSpan(p_off).subspan(1, N-1)
  ));

  // [Constraint] All-gather communication time during module i
  for (int i = 0; i < N; i++) {
    model.AddDivisionEquality(ag[i], p_sha[i], D2D_BW);
  }
  ag.push_back(model.NewConstant(0));

  // [Constraint] Reduce-scatter communication time during module i
  // for (int i = 0; i < N; i++) {
  //   model.AddDivisionEquality(rs[i], p_sha[i], D2D_BW);
  // }

  // [Constraint] Reload communication time during module i
  for (int i = 0; i < N; i++) {
    model.AddDivisionEquality(rd[i], p_off[i], H2D_BW);
  }
  rd.push_back(model.NewConstant(0));

  // Link between the wrapped unit and next wrapped unit in forward
  // fw_index[i] == j where wrap[j] is the first wrap variable set to true after i
  std::vector<IntVar> fw_index(N, model.NewConstant(N));
  for (int i = 0; i < N-1; i++) {
    fw_index[i] = model.NewIntVar(Domain(0, N+1)).WithName(absl::StrCat("fw_index_", i));
    std::vector<LinearExpr> exprs;
    for (int j = i + 1; j < N; j++) {
      exprs.push_back(N - wrap[j] * (N - j));
    }
    model.AddMinEquality(fw_index[i], exprs);
  }
  // Redundant constraint
  for (int i = 0; i < N-1; i++) {
    model.AddLessOrEqual(fw_index[i], fw_index[i+1]);
  }

  // Link between the wrapped unit and next wrapped unit in backward
  // bw_index[i] == j where wrap[j] is the first wrap variable set to true after i 
  std::vector<IntVar> bw_index(N, model.NewConstant(N));
  for (int i = 0; i < N-1; i++) {
    int bi = pos2pre[i];
    bw_index[bi] = model.NewIntVar(Domain(0, N+1)).WithName(absl::StrCat("bw_index_", i));
    std::vector<LinearExpr> exprs;
    for (int j = i + 1; j < N; j++) {
      int bj = pos2pre[j];
      exprs.push_back((N + 1) - wrap[bj] * (N + 1 - bj));
    }
    model.AddMinEquality(bw_index[bi], exprs);
  }
  // Redundant constraint
  for (int i = 0; i < N-1; i++) {
    int bi = pos2pre[i];
    int bj = pos2pre[i+1];
    model.AddLessOrEqual(bw_index[bi], bw_index[bj]);
  }

  // [Constraint] Forward prefetch all-gather communication time
  for (int i = 0; i < N; i++) {
    IntVar fw_ag_i = model.NewIntVar(Domain(0, D2D_UB)).WithName(absl::StrCat("fw_ag_", i));
    model.AddVariableElement(fw_index[i], ag, fw_ag_i);
    model.AddEquality(fw_ag[i], fw_ag_i).OnlyEnforceIf(wrap[i]);
    model.AddEquality(fw_ag[i], 0).OnlyEnforceIf(wrap[i].Not());
  }

  // [Constraint] Backward prefetch all-gather communication time
  for (int i = 0; i < N; i++) {
    IntVar bw_ag_i = model.NewIntVar(Domain(0, D2D_UB)).WithName(absl::StrCat("bw_ag_", i));
    model.AddVariableElement(bw_index[i], ag, bw_ag_i);
    model.AddEquality(bw_ag[i], bw_ag_i).OnlyEnforceIf(wrap[i]);
    model.AddEquality(bw_ag[i], 0).OnlyEnforceIf(wrap[i].Not());
  }

  // [Constraint] Previous module's reduce scatter communication time
  for (int i = 0; i < N; i++) {
    IntVar bw_rs_i = model.NewIntVar(Domain(0, D2D_UB)).WithName(absl::StrCat("bw_rs_", i));
    model.AddVariableElement(fw_index[i], ag, bw_rs_i);
    model.AddEquality(bw_rs[i], bw_rs_i).OnlyEnforceIf(wrap[i]);
    model.AddEquality(bw_rs[i], 0).OnlyEnforceIf(wrap[i].Not());
  }

  // [Constraint] Forward preload communication overhead
  for (int i = 0; i < N; i++) {
    IntVar fw_rd_i = model.NewIntVar(Domain(0, H2D_UB)).WithName(absl::StrCat("fw_rd_", i));
    model.AddVariableElement(fw_index[i], rd, fw_rd_i);
    model.AddEquality(fw_rd[i], fw_rd_i).OnlyEnforceIf(wrap[i]);
    model.AddEquality(fw_rd[i], 0).OnlyEnforceIf(wrap[i].Not());
  }

  // [Constraint] Backward preload communication overhead
  for (int i = 0; i < N; i++) {
    IntVar bw_rd_i = model.NewIntVar(Domain(0, H2D_UB)).WithName(absl::StrCat("bw_rd_", i));;
    model.AddVariableElement(bw_index[i], rd, bw_rd_i);
    model.AddEquality(bw_rd[i], bw_rd_i).OnlyEnforceIf(wrap[i]);
    model.AddEquality(bw_rd[i], 0).OnlyEnforceIf(wrap[i].Not());
  }

  // [Constraint] Communication overhead in forward
  // This is a linearization for the non-linear constraints:
  //   c_fw[i] == max(0, fw_ag[i] - F[i], fw_rd[i] - F[i]) * wrap[i]
  model.AddEquality(c_fw[0], 0);
  for (int i = 1; i < N; i++) {
    auto fw_e = model.NewIntVar(Domain(0, std::max(H2D_UB, D2D_UB))).WithName(absl::StrCat("fw_e_", i));
    model.AddMaxEquality(fw_e, {fw_ag[i] - F[i], fw_rd[i] - F[i], 0});
    model.AddEquality(c_fw[i], bool_int_prod(model, wrap[i], fw_e));
  }

  // [Constraint] Communication overhead in backward
  // This is a linearization for the non-linear constraints:
  //   c_bw[i] == max(0, bw_ag[i] - B[i], bw_rd[i] - B[i], bw_rs[i] - B[i]) * wrap[i]
  model.AddEquality(c_bw[0], 0);
  for (int i = 1; i < N; i++) {
    auto bw_e = model.NewIntVar(Domain(0, std::max(H2D_UB, D2D_UB))).WithName(absl::StrCat("bw_e_", i));
    auto b = model.NewIntVar(Domain(0, std::max(H2D_UB, D2D_UB))).WithName(absl::StrCat("b_", i));
    model.AddEquality(b, bw_ag[i] + bw_rs[i]);
    model.AddMaxEquality(bw_e, {b - B[i], bw_rd[i] - B[i], 0});
    model.AddEquality(c_bw[i], bool_int_prod(model, wrap[i], bw_e));
  }

  // Communication overheads in forward and backward of non-root modules
  LinearExpr submodule_overhead;
  for (int i = 1; i < N; i++) {
    submodule_overhead += c_fw[i] + c_bw[i];
  }

  // Cmmunication overheads for the root module, as they cannot 
  // be overlapped with other modules
  LinearExpr root_overhead = ag[0] + rs[0] + fw_ag[0] + bw_ag[0] + bw_rs[0] + rd[0] + fw_rd[0] + bw_rd[0];

  // Objective
  model.Minimize(submodule_overhead + root_overhead + rcp);

  parameters.set_relative_gap_limit(0.05);
  resp = SolveWithParameters(model.Build(), parameters);

  LOG(INFO) << absl::StrFormat("Stage 2 completed (status: %s, objective: %f, wall-time (s): %f)", 
    CpStatusString(resp.status()), resp.objective_value(), resp.wall_time());

  auto result = CpSolverResult();
  if (resp.status() == CpSolverStatus::OPTIMAL || resp.status() == CpSolverStatus::FEASIBLE) {
    result.objective = resp.objective_value();
    result.peak_device_memory = SolutionIntegerValue(resp, m_gpu);

    for (auto &[n, v]: named_parameters) {
      result.param_policy[n] = std::make_tuple(
        SolutionBooleanValue(resp, x[n]),
        SolutionBooleanValue(resp, y[n])
      );
    }

    for (int i = 0; i < N; ++i) {
      if (SolutionBooleanValue(resp, ckpt[i])) {
        result.modules_to_recompute.insert(i);
      }
      if (SolutionBooleanValue(resp, wrap[i])) {
        result.modules_to_wrap.insert(i);
      }
    }

    // debugging
    if (DEBUG) {
      std::cout << absl::StrCat(
        "total_sharded = ", SolutionIntegerValue(resp, total_sharded), ", ",
        "total_replicated = ", SolutionIntegerValue(resp, total_replicated), ", ",
        "total_offloaded = ", SolutionIntegerValue(resp, total_offloaded), ", ",
        "total_sharded_per_dp = ", SolutionIntegerValue(resp, total_sharded_per_dp)
      ) << std::endl;
      std::cout << absl::StrCat("max_a = ", SolutionIntegerValue(resp, max_a)) << std::endl;
      for (int i = 0; i < N; i++) {
        std::cout << absl::StrFormat(
          "%d, wrap=%d, p_sha = %d, p_off = %d, p_rep = %d, ag = %d, rs = %d, rd = %d, fw_ag = %d, fw_rd = %d, bw_ag = %d, bw_rs = %d, bw_rd = %d",
          i,
          SolutionBooleanValue(resp, wrap[i]),
          SolutionIntegerValue(resp, p_sha[i]),
          SolutionIntegerValue(resp, p_off[i]),
          SolutionIntegerValue(resp, p_rep[i]),
          SolutionIntegerValue(resp, ag[i]),
          SolutionIntegerValue(resp, rs[i]),
          SolutionIntegerValue(resp, rd[i]),
          SolutionIntegerValue(resp, fw_ag[i]),
          SolutionIntegerValue(resp, fw_rd[i]),
          SolutionIntegerValue(resp, bw_ag[i]),
          SolutionIntegerValue(resp, bw_rs[i]),
          SolutionIntegerValue(resp, bw_rd[i])
        ) << std::endl;
      }
      for (auto &[n, p]: named_parameters) {
        std::cout << absl::StrFormat("%s: x=%d, y=%d", n,
          SolutionBooleanValue(resp, x[n]), SolutionBooleanValue(resp, y[n])) << std::endl;
      }
    }

    return result;
  }

  return std::nullopt;
}

}  // namespace sat
}  // namespace operations_research


PYBIND11_MODULE(pybind_solver, m) {
  m.def("call_cp_solver", &operations_research::sat::CallCpSolver);

  py::class_<operations_research::sat::CpSolverResult>(m, "CpSolverResult")
    .def_readwrite("param_policy", &operations_research::sat::CpSolverResult::param_policy)
    .def_readwrite("modules_to_wrap", &operations_research::sat::CpSolverResult::modules_to_wrap)
    .def_readwrite("modules_to_recompute", &operations_research::sat::CpSolverResult::modules_to_recompute)
    .def_readwrite("peak_device_memory", &operations_research::sat::CpSolverResult::peak_device_memory)
    .def_readwrite("objective", &operations_research::sat::CpSolverResult::objective);
}