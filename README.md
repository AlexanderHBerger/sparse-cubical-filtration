# Sparse cubical complexes for efficient topology-preservation in image data

Alexander H. Berger<sup>1,2</sup>, Marco Fontana<sup>2</sup>, Daniel Rueckert<sup>2,3,4</sup>,
Johannes C. Paetzold<sup>1</sup>, Laurin Lux<sup>2</sup>, Ulrich Bauer<sup>2,4,5</sup>

<sup>1</sup> Cornell University<br>
<sup>2</sup> Technical University of Munich<br>
<sup>3</sup> Department of Computing, Imperial College London, UK<br>
<sup>4</sup> Munich Center for Machine Learning (MCML), Munich, Germany<br>
<sup>5</sup> Munich Data Science Institute, Technical University of Munich, Munich, Germany

This repository contains our implementation of sparse cubical complexes and of
**sparseBM**, a topological loss function for image segmentation. Sparse cubical
complexes omit the cells of confident background regions, i.e., voxels where both the
prediction and the label are background. Each connected background region is replaced
by a single virtual node. This makes persistent homology on common training patch sizes
fast enough for network training.

- `cpp/`: persistent homology and Betti matching on sparse cubical complexes (2D and 3D)
  with Python bindings (module `betti_matching`). It extends the Betti matching
  implementation of [Stucki et al. (2024)](https://arxiv.org/abs/2407.04683).
- `sparse_bm/`: the sparseBM loss in PyTorch.

## Installation

Requirements: a C++17 compiler, CMake >= 3.5, Python >= 3.9 with `pybind11`, `numpy`,
`scipy` and `torch`. `cupy` is optional and computes the pocket labels on the GPU.

```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j betti_matching
cd ../..
export PYTHONPATH=$PWD/cpp/build:$PWD:$PYTHONPATH
```

Adding `-DCMAKE_CXX_FLAGS=-DVALUE_T_FLOAT` to the `cmake` call builds a float32 version,
which needs half the memory. The Python package works with both builds.

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

`penalty_matched` and `penalty_unmatched` select squared error on the filtration values
(`"se"`) or cross-entropy on the logits (`"ce"`, requires `logits`). `clamp_confident`
(default 0.99, 1.0 disables it) sets confident foreground values to a single value in the
input of the matching, which makes the matching faster; the loss itself uses the exact
values. The matching runs on the CPU with one thread per batch element. To run it in
parallel to GPU work, call `topo.build_masks(p_fg, target)` first and
`topo.loss_from_masks(p_fg, *masks, logits=logits)` later.

The matching can also be computed directly:

```python
import betti_matching
from sparse_bm import make_sparse_pair, label_pockets

m_pred, m_target, keep = make_sparse_pair(g_pred, g_target, tau)   # g = 1 - p
res = betti_matching.compute_matching(m_pred, m_target, sparse=True, mask_threshold=tau,
                                      pocket_labels=label_pockets(keep).fine)
```

Bars that are still alive when the background enters are returned in the
`*_essential_birth_coordinates` fields of the result.

## License

MIT, see `LICENSE`. The code in `cpp/` builds on Betti-Matching-3D, see
`cpp/LICENSE.upstream`.
