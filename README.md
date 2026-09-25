# Sparse Betti Matching

Code for the loss function of the submission *"Sparse Cubical Complexes for Efficient
Topology-Preservation in Image Data"*: Betti matching computed on a **sparse cubical
complex**, and the resulting topological segmentation loss (**sparseBM**).

- `cpp/`: C++ persistence and Betti matching with Python bindings (module
  `betti_matching`), extending the Betti-Matching-3D implementation (Stucki et al., 2024)
  with the sparse complex (2D and 3D).
- `sparse_bm/`: Python package (masks, pocket labels, loss terms, loss module).

## Installation

Requirements: a C++17 compiler, CMake >= 3.5, Python >= 3.9 with `pybind11`, `numpy`,
`scipy` and `torch`. Optional: `cupy` (pocket labelling on the GPU).

```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j betti_matching
cd ../..
export PYTHONPATH=$PWD/cpp/build:$PWD:$PYTHONPATH
```

Adding `-DCMAKE_CXX_FLAGS=-DVALUE_T_FLOAT` to the `cmake` call builds a float32 variant
with half the memory footprint; the Python package adapts to either build.

## Usage

```python
import torch
from sparse_bm import SparseBettiLossBatched

topo = SparseBettiLossBatched(tau_mask=0.8, penalty_matched="se", penalty_unmatched="se")

logits = model(x)                        # (B, 2, *spatial), 2D or 3D
p_fg = torch.softmax(logits, dim=1)[:, 1]
target = (y == 1).float()                # (B, *spatial), binary
loss = dice_ce(logits, y) + weight * topo(p_fg, target, logits=logits)
loss.backward()
```

`penalty_matched` / `penalty_unmatched` select squared error on the filtration values
(`"se"`) or cross-entropy on the logits (`"ce"`, requires `logits`). `clamp_confident`
(default 0.99, 1.0 disables it) collapses confident foreground values in the matcher's copy
of the input, which makes the matching cheaper; the loss terms always read the exact
values. The matching runs on the CPU with one thread per batch element; to overlap it with
GPU work, call `topo.build_masks(p_fg, target)` and later
`topo.loss_from_masks(p_fg, *masks, logits=logits)`.

The matcher can also be called directly:

```python
import betti_matching
from sparse_bm import make_sparse_pair, label_pockets

m_pred, m_target, keep = make_sparse_pair(g_pred, g_target, tau)   # g = 1 - p, +inf off the mask
res = betti_matching.compute_matching(m_pred, m_target, sparse=True, mask_threshold=tau,
                                      pocket_labels=label_pockets(keep).fine)
```

Censored bars of a sparse matching are returned in the `*_essential_birth_coordinates`
fields of the result.
