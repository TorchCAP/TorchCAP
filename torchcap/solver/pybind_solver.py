import os
from pathlib import Path

solver = None

def load():
    global solver
    if solver is None:
        from torch.utils.cpp_extension import load as jit_load
        base_path = os.path.join(Path(__file__).parent.parent.absolute())
        solver = jit_load(
            name='pybind_solver',
            sources=[f'{base_path}/compactron/pybind_solver.cc'],
            extra_cflags=['-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
            extra_include_paths=[f'{base_path}/third_party/ortools/include'],
            extra_ldflags=[
                f'-L{os.path.abspath(f'{base_path}/third_party/ortools/lib')}',
                '-lortools',
            ],
            verbose=True,
        )
    return solver
