#pragma once

#include "../config.h"

#include <cstddef> // For std::ptrdiff_t
#include <cstdint>
#include <iostream>
#include <iterator> // For std::forward_iterator_tag
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <sstream>

using namespace std;

namespace dim3 {
typedef tuple<index_t, index_t, index_t> Coordinate;
typedef vector<Coordinate> RepresentativeCycle;

#ifdef RUNTIME
// Init-phase timing attribution (RUNTIME builds only).
extern double g_tGrid, g_tKept, g_tPocketVal, g_tPocketTop,
    g_tConeVE, g_tConeFaces, g_tConeInc, g_tConeEmit, g_tConeCycles,
    g_tRangeCheck;
#endif

class Cube {
  public:
    Cube();
    Cube(value_t birth, index_t x, index_t y, index_t z, uint8_t type);
    Cube(value_t birth, vector<index_t> coordinates, uint8_t type);
    Cube(const Cube &cube);
    bool operator==(const Cube &rhs) const;
    index_t x() const;
    index_t y() const;
    index_t z() const;
    uint8_t type() const;
    void print() const;
    value_t birth;
    uint64_t index;
};

struct CubeComparator {
    bool operator()(const Cube &Cube1, const Cube &Cube2) const;
};

class Pair {
  public:
    Pair();
    Pair(const Cube &birth, const Cube &death);
    Pair(const Pair &pair);
    bool operator==(const Pair &rhs) const;
    void print() const;
    Cube birth;
    Cube death;
};

class Match {
  public:
    Match(Pair pair0, Pair pair1);
    void print() const;
    Pair pair0;
    Pair pair1;
};

class CubicalGridComplex {
  public:
    CubicalGridComplex(const vector<value_t> &image,
                       const vector<index_t> &shape);
    CubicalGridComplex(CubicalGridComplex &&other);
    ~CubicalGridComplex();
    size_t getNumberOfCubes(const uint8_t &dim) const;
    value_t getBirth(const index_t &x, const index_t &y,
                     const index_t &z) const;
    value_t getBirth(const index_t &x, const index_t &y, const index_t &z,
                     const uint8_t &type, const uint8_t &dim) const;
    Coordinate getParentVoxel(const Cube &c, const uint8_t &dim) const;
    void printImage() const;
    void printRepresentativeCycle(const RepresentativeCycle &reprCycle) const;
    const vector<index_t> shape;
    const index_t m_x;
    const index_t m_y;
    const index_t m_z;
    const index_t m_yz;
    const index_t m_xyz;
    const index_t n_yz;
    const index_t n_xyz;

  private:
    void getGridFromVector(const vector<value_t> &vector);
    // Contiguous padded (shape+2)^3 grid, row-major. getBirth is one
    // arithmetic load rather than three dependent pointer dereferences, and
    // construction is a single allocation instead of ~O((s+2)^2) new[] calls.
    vector<value_t> grid;
    size_t gridStrideX;
    size_t gridStrideY;
};

// Kept-cell reindexing: one O(N) pass over the
// comparison complex classifies every cell as kept (finite birth) or masked.
// Masked voxels are INFTY in BOTH inputs (shared mask), so a cell has finite
// birth in the comparison complex iff it does in input 0 iff in input 1 --
// this single classification serves all three complexes and both image
// complexes. Per cell dimension it provides
//   - a dense int32 compact-id array over the SAME slot space the dense
//     structures use (vertices: the dim-0 UnionFind vertex ids; edges and
//     2-cells: the CubeMap (x,y,z,type) slots; 3-cells: the UnionFindDual
//     node ids), -1 marking masked/absent cells, and
//   - the kept-cell list in the SAME canonical nested-loop order
//     (x,y,z[,type]) as the full-grid enumerations, so downstream stable
//     sorts and processing orders see identical sequences.
// A kept cell may have a finite birth >= maskThreshold in one input (it is
// kept because the other input is below threshold there).
class KeptCells {
  public:
    explicit KeptCells(const CubicalGridComplex &cgcComp);

    int32_t vertexCompact(index_t vertexIdx) const {
        return compactVertex[vertexIdx];
    }
    int32_t edgeCompact(uint64_t cubeIndex) const {
        return compactEdge[slotIndex(cubeIndex)];
    }
    int32_t twoCellCompact(uint64_t cubeIndex) const {
        return compactTwoCell[slotIndex(cubeIndex)];
    }
    int32_t topCellCompact(index_t dualNodeIdx) const {
        return compactTopCell[dualNodeIdx];
    }

    // Kept lists in canonical order: dense vertex ids (x*n_yz + y*n_z + z),
    // packed cube spatial keys (x<<44 | y<<24 | z<<4 | type), and dual node
    // ids (x*m_yz + y*m_z + z) respectively.
    vector<index_t> keptVertices;
    vector<uint64_t> keptEdges;
    vector<uint64_t> keptTwoCells;
    vector<index_t> keptTopCells;

  private:
    // Same slot math as CubeMap<1/2,_Tp>::computeCoordinateIndex.
    size_t slotIndex(uint64_t cubeIndex) const {
        const size_t x = (cubeIndex >> 44) & 0xfffff;
        const size_t y = (cubeIndex >> 24) & 0xfffff;
        const size_t z = (cubeIndex >> 4) & 0xfffff;
        const size_t type = cubeIndex & 0xf;
        return x * strideX + y * strideY + z * 3 + type;
    }
    vector<int32_t> compactVertex;
    vector<int32_t> compactEdge;
    vector<int32_t> compactTwoCell;
    vector<int32_t> compactTopCell;
    size_t strideX;
    size_t strideY;
};

// Sparse mode: the virtual nodes that stand in for the masked region, which
// enters as one terminal slice at CONE_BIRTH. See src_2D/data_structures.h for
// the full argument; the 3D version is identical with 8 corners instead of 4.
class PocketNodes {
  public:
    // `voxelLabels` MUST outlive this object: the labels are not copied and
    // ofTopCellAt() reads them on demand. Materialising the cube-label array
    // instead costs an int32[m_xyz] and an 8N-load
    // O(N) pass, for a table that is only ever probed O(kept) times -- twice
    // per kept 2-cell in the dual UF and <=8 per rim cell in ConeIndex.
    PocketNodes(const CubicalGridComplex &cgcComp,
                const vector<int32_t> &voxelLabels, size_t keptVoxels);

    int32_t numPockets() const { return n; }
    // Pocket of the top cell with min-corner (cx,cy,cz); 0 if kept. The cube
    // is non-kept iff a corner is masked, and its pocket is the max over the
    // corner labels -- exact because all masked corners of one cube lie within
    // L_inf <= 1, hence in one fine pocket.
    int32_t ofTopCellAt(index_t cx, index_t cy, index_t cz) const {
        if (!cubePocket.empty()) {
            return cubePocket[static_cast<size_t>(cx) * myz +
                              static_cast<size_t>(cy) * mz + cz];
        }
        const int32_t *L = labels->data();
        const size_t base = static_cast<size_t>(cx) * syz +
                            static_cast<size_t>(cy) * sz + cz;
        int32_t best = 0;
        for (size_t dx = 0; dx <= syz; dx += syz) {
            for (size_t dy = 0; dy <= sz; dy += sz) {
                const int32_t a = L[base + dx + dy];
                const int32_t b = L[base + dx + dy + 1];
                if (a > best) { best = a; }
                if (b > best) { best = b; }
            }
        }
        return best;
    }
    // Flat dual-node-id form. Costs two integer divisions -- prefer
    // ofTopCellAt on anything hot.
    int32_t ofTopCell(index_t nodeIdx) const {
        return ofTopCellAt(nodeIdx / myz, nodeIdx / mz % my, nodeIdx % mz);
    }

    // True iff the pocket contains a masked voxel on a grid face, i.e. it IS
    // the exterior and must be pre-unioned with the sentinel.
    bool touchesBoundary(int32_t pocket) const { return boundary[pocket] != 0; }

  private:
    const vector<int32_t> *labels = nullptr;
    // Optional memo of the cube labelling. Materialising it costs 8N label
    // loads plus an int32[m_xyz]; probing lazily costs 8 loads per probe, and
    // the probes are O(kept): ~8 per rim vertex, 4 per rim edge, 2 per rim
    // face, 2 per kept 2-cell in each dual UF -- call it ~26 per kept voxel.
    // So the memo pays iff 26 * kept > N.
    vector<int32_t> cubePocket;
    size_t syz = 0;                 // voxel-grid strides for ofTopCellAt
    size_t sz = 0;
    index_t mx = 0, my = 0, mz = 0; // cube-grid extents for ofTopCell
    size_t myz = 0;
    int32_t n = 0;
    vector<uint8_t> boundary;
};

// Virtual (non-grid) cells of the sparse dim-1 construction, encoded in
// the unused bits of Cube::index's 4-bit `type` field: the grid uses 0..2, and
// 3..14 are free. Coordinates stay meaningful -- a virtual cell is anchored at
// the kept cell it is coned from -- so CubeComparator, the (birth, index)
// tie-break and every debug print keep working, and virtual cells sort AFTER
// real ones at the same coordinate, which is what the boundary order needs.
//
//   coneEdge(p, v)    type = 3 + rankV      rankV in [0, 8)  -> type 3..10
//   cone2Cell(p, e)   type = 3 + 4*e.type + rankE   rankE in [0, 4) -> 3..14
//
// where rankV/rankE is the position of pocket p in the SORTED list of distinct
// pockets touching that cell. That keeps the encoding canonical (it is
// derived from the shared mask, so it is identical in I, J and C) and bounded.
constexpr uint8_t VIRTUAL_TYPE_BASE = 3;
inline bool isVirtualCell(uint64_t index) {
    return (index & 0xf) >= VIRTUAL_TYPE_BASE;
}

// Sparse dim-1 incidence: which pockets each kept vertex / edge is a rim
// cell of. A kept cell is a rim cell of p iff some non-kept CUBE of p has
// it as a face -- note this is the CUBE stencil, not the smaller stencil of
// masked cofacet 2-cells: a cube can be non-kept because a corner diagonally
// opposite the edge is masked while both of its 2-cells containing that edge
// are kept.
//
// The chain complex closes because an edge's cube stencil is contained in each
// endpoint's, so every vertex of a rim edge of p is a rim vertex of p:
//     d(cone2Cell(p, e)) = e + coneEdge(p, v0) + coneEdge(p, v1).
class ConeIndex {
  public:
    ConeIndex(const CubicalGridComplex &cgcComp, const KeptCells &kept,
              const PocketNodes &pockets, bool prune = false);

    size_t numConeEdges() const { return vPocket.size(); }
    size_t numCone2Cells() const { return ePocket.size(); }

    // Compact row/column ids, laid out after the kept cells of that dimension.
    int32_t coneEdgeSlot(int32_t keptVertex, uint8_t rank) const {
        return vStart[keptVertex] + rank;
    }
    int32_t cone2CellSlot(int32_t keptEdge, uint8_t rank) const {
        return eStart[keptEdge] + rank;
    }
    uint8_t coneEdgeCount(int32_t keptVertex) const {
        return static_cast<uint8_t>(vStart[keptVertex + 1] - vStart[keptVertex]);
    }
    uint8_t cone2CellCount(int32_t keptEdge) const {
        return static_cast<uint8_t>(eStart[keptEdge + 1] - eStart[keptEdge]);
    }
    int32_t coneEdgePocket(int32_t keptVertex, uint8_t rank) const {
        return vPocket[vStart[keptVertex] + rank];
    }
    int32_t cone2CellPocket(int32_t keptEdge, uint8_t rank) const {
        return ePocket[eStart[keptEdge] + rank];
    }
    // Precomputed ranks of the same pocket at the edge's two endpoints, so
    // enumerating a cone 2-cell's boundary is O(1) with no search.
    uint8_t rankAtV0(int32_t keptEdge, uint8_t rank) const {
        return eRankV0[eStart[keptEdge] + rank];
    }
    uint8_t rankAtV1(int32_t keptEdge, uint8_t rank) const {
        return eRankV1[eStart[keptEdge] + rank];
    }

    // --- pruning (config.conePrune) --------------------------------------
    // Emit cone2Cell for a slot iff its bit is set. Slot NUMBERING is
    // unchanged by pruning -- suppressed columns leave GAPS -- so every
    // pivotColumnIndex / SparseOrDenseCubeMap sizing and every
    // cone2CellSlot() stays valid, and the (birth, index) tie-break inside
    // the virtual slice is not reshuffled. Compacting the slots instead would
    // renumber every surviving column's Cube index and change which cells
    // are reported as deaths inside the virtual slice, so it is deliberately
    // not done.
    bool emitCone2Cell(int32_t slot) const {
        return (emitBits[static_cast<size_t>(slot) >> 6] >>
                (static_cast<size_t>(slot) & 63)) &
               1ULL;
    }
    size_t numEmittedCone2Cells() const { return nEmitted; }
    // Rim 2-cells, as (kept 2-cell, pocket) pairs. Only pruning needs these.
    size_t numRimFaces() const { return fPocket.size(); }

    // --- explicit fundamental cycles --------------------------------------
    // The emitted cone block is a spanning FOREST of the rim graph (whose
    // vertices are cone edges and whose edges are cone 2-cells, one connected
    // piece per pocket) plus b_1 non-tree columns. Reducing it by Gaussian
    // elimination just recomputes that forest: a tree column claims a cone
    // edge at CONE_BIRTH, which is its own birth, so it emits NOTHING, and a
    // non-tree column only becomes useful once its two cone edges have cancelled
    // against the tree columns along the path between its endpoints.
    //
    // So the forest is built here instead, with a union-find over the emitted
    // slots in SLOT order -- which is exactly the (birth, index) order the
    // reduction sees, because slot order is (x, y, z, edge type, rank) and a
    // cone 2-cell's Cube index is (x, y, z, 3 + 4*type + rank). Every non-tree
    // column's endpoints are therefore already joined by STRICTLY EARLIER tree
    // columns, so its fundamental cycle is a combination of earlier columns --
    // which is what makes the substitution rank-preserving, hence pairing- and
    // output-identical.
    //
    // What is emitted is then one column per non-tree slot, carrying the
    // explicit cycle `e + path_T(u, v)` over REAL kept edges only. No cone
    // edge and no tree column is ever built.
    bool hasCycles() const { return !cycPtr.empty(); }
    bool emitCycleColumn(int32_t slot) const {
        return cycIndex[static_cast<size_t>(slot)] >= 0;
    }
    int32_t cycleLength(int32_t slot) const {
        const int32_t c = cycIndex[static_cast<size_t>(slot)];
        return cycPtr[c + 1] - cycPtr[c];
    }
    // Faces in DECREASING Cube-index order, which is what hasPreviousFace and
    // the emergent-pair lemma need: the faces are all real kept edges, and the
    // maximal one under (birth, index) among those at the column's birth is
    // the first one reached scanning indices downwards.
    uint64_t cycleEdge(int32_t slot, int32_t k) const {
        return cycEdge[static_cast<size_t>(
                           cycPtr[cycIndex[static_cast<size_t>(slot)]]) +
                       static_cast<size_t>(k)];
    }
    size_t numCycleColumns() const {
        return cycPtr.empty() ? 0 : cycPtr.size() - 1;
    }
    size_t numCycleEntries() const { return cycEdge.size(); }

  private:
    vector<int32_t> vStart, vPocket;
    vector<int32_t> eStart, ePocket;
    vector<uint8_t> eRankV0, eRankV1;
    // Rim faces in the same CSR shape, plus rim-EDGE slot -> rim-FACE slots.
    vector<int32_t> fStart, fPocket;
    vector<int32_t> efStart, efFace;
    vector<uint64_t> emitBits;
    size_t nEmitted = 0;
    // cone2Cell slot -> index into cycPtr, or -1 (tree column / not emitted).
    vector<int32_t> cycIndex;
    vector<int32_t> cycPtr;
    vector<uint64_t> cycEdge;   // packed kept-edge keys, DESCENDING per cycle

    void buildFaces(const CubicalGridComplex &cgcComp, const KeptCells &kept,
                    const PocketNodes &pockets);
    void buildEdgeFaceIncidence(const CubicalGridComplex &cgcComp,
                                const KeptCells &kept);
    void computeEmitMask(const CubicalGridComplex &cgcComp,
                         const KeptCells &kept, bool prune);
    void buildCycles(const CubicalGridComplex &cgcComp, const KeptCells &kept);
};

class UnionFind {
  public:
    // kept == nullptr: dense over all vertices (bit-identical to upstream).
    // kept != nullptr (sparse mode): compact over the kept vertices
    // only; node ids are compact ids, and equal-birth tie-breaks compare the
    // stored ORIGINAL dense vertex indices (never compact ids).
    UnionFind(const CubicalGridComplex &cgc, const KeptCells *kept = nullptr);
    index_t find(index_t x);
    index_t link(index_t x, index_t y);
    value_t getBirth(const index_t &idx) const;
    Coordinate getCoordinates(index_t idx) const;
    vector<index_t> getBoundaryIndices(const Cube &edge) const;
    index_t size() const { return static_cast<index_t>(parent.size()); }
    void reset();

  private:
    vector<index_t> parent;
    vector<value_t> birthtime;
    // Compact (sparse) mode only: compact id -> original dense vertex id
    // (strictly increasing; empty in dense mode).
    vector<index_t> original;
    const CubicalGridComplex &cgc;
    const KeptCells *kept;
};

class UnionFindDual {
  public:
    // kept == nullptr: dense over all top-cells + exterior sentinel
    // (bit-identical to upstream). kept != nullptr (sparse mode):
    // compact over the kept top-cells, the exterior sentinel as the LAST
    // node (compact id size()-1, original id m_xyz); equal-birth tie-breaks
    // compare original dense node ids.
    // Sparse mode adds one node per pocket AFTER the exterior sentinel, all
    // born at `pocketBirth` (= CONE_BIRTH). Node layout is then
    //     [kept top cells 0 .. K-1] [exterior sentinel K] [pockets K+1 .. K+P]
    // so `K + pocketId` addresses a pocket and `K + 0` the sentinel, making
    // the out-of-grid and masked cases one uniform lookup.
    UnionFindDual(const CubicalGridComplex &cgc,
                  const KeptCells *kept = nullptr,
                  const PocketNodes *pockets = nullptr,
                  value_t pocketBirth = INFTY);
    // A virtual pocket node has no voxel: a bar whose dying root is one is
    // CENSORED (death = CONE_BIRTH) and must be reported by its birth cell
    // only.
    bool isPocketNode(index_t idx) const {
        return pockets != nullptr && idx > sentinelIdx;
    }
    index_t sentinel() const { return sentinelIdx; }
    // Pre-union every boundary-touching pocket into the sentinel. Must run
    // after each reset(), before the sweep.
    void seedPockets();
    index_t find(index_t x);
    index_t link(index_t x, index_t y);
    value_t getBirth(const index_t &idx) const;
    Coordinate getCoordinates(index_t idx) const;
    vector<index_t> getBoundaryIndices(const Cube &edge) const;
    index_t size() const { return static_cast<index_t>(parent.size()); }
    void reset();

  private:
    vector<index_t> parent;
    vector<value_t> birthtime;
    // Compact (sparse) mode only: compact id -> original dense dual node id.
    vector<index_t> original;
    const CubicalGridComplex &cgc;
    const KeptCells *kept;
    const PocketNodes *pockets = nullptr;
    index_t sentinelIdx = 0;
};

template <int _dim, class _Tp> class CubeMap {
    /// Datastructure to map cube indices to values efficiently via a
    /// 1-dimensional array by mapping cube indices to (x,y,z,type) coordinates
    /// in a 4-dimensional space and representing this as a 1-dimensional array.

  public:
    CubeMap(vector<index_t> shape);
    void emplace(uint64_t cube_index, _Tp element);
    const std::optional<_Tp> &find(uint64_t cube_index) const;
    void clear();
    optional<_Tp> &operator[](uint64_t cube_index);

  private:
    vector<std::optional<_Tp>> elements;
    uint64_t computeCoordinateIndex(uint64_t cube_index) const;
    vector<index_t> shape;
    std::optional<_Tp> none = {};
    static const int NUM_TYPES = (_dim == 1 || _dim == 2) ? 3 : 1;
    const int stride_x_direction;
    const int stride_y_direction;
    const int stride_z_direction;
};

template <int _dim, class _Tp>
CubeMap<_dim, _Tp>::CubeMap(vector<index_t> shape)
    : shape(shape), stride_x_direction(shape[1] * shape[2] * NUM_TYPES),
      stride_y_direction(shape[2] * NUM_TYPES), stride_z_direction(NUM_TYPES),
      elements(shape[0] * shape[1] * shape[2] * NUM_TYPES) {}

template <int _dim, class _Tp>
void CubeMap<_dim, _Tp>::emplace(uint64_t cube_index, _Tp element) {
    if (cube_index != NONE_INDEX) {
        elements[computeCoordinateIndex(cube_index)] = element;
    } else {
        throw runtime_error(
            "CubeMap::emplace may not be called with NONE_INDEX magic number");
    }
}

template <int _dim, class _Tp>
const std::optional<_Tp> &CubeMap<_dim, _Tp>::find(uint64_t cube_index) const {
    if (cube_index != NONE_INDEX) {
        return elements[computeCoordinateIndex(cube_index)];
    }
    return none;
}

template <int _dim, class _Tp>
optional<_Tp> &CubeMap<_dim, _Tp>::operator[](uint64_t cube_index) {
    if (cube_index != NONE_INDEX) {
        return elements[computeCoordinateIndex(cube_index)];
    }
    throw runtime_error("CubeMap subscript operator may not be called with "
                        "NONE_INDEX magic number");
}

template <int _dim, class _Tp>
uint64_t CubeMap<_dim, _Tp>::computeCoordinateIndex(uint64_t cube_index) const {
    int x = (cube_index >> 44) & 0xfffff;
    int y = (cube_index >> 24) & 0xfffff;
    int z = (cube_index >> 4) & 0xfffff;
    int type = (_dim == 1 || _dim == 2) ? (cube_index & 0xf) : 0;

    return x * stride_x_direction + y * stride_y_direction +
           z * stride_z_direction + type;
}

template <int _dim, class _Tp> void CubeMap<_dim, _Tp>::clear() {
    elements.clear();
    elements.resize(shape[0] * shape[1] * shape[2] * NUM_TYPES);
}

// CubeMap facade. Constructed without a kept index it contains a real
// dense CubeMap (bit-identical to upstream); constructed with one (sparse
// mode) it stores a K-sized compact payload indexed through the kept-cell
// compact ids. `find` on a cell without a compact id returns none; writing
// such a cell is an sparse invariant violation and throws. _dim selects which
// kept list backs the payload and must match the KEY space of the map
// (edge-keyed maps: 1, 2-cell-keyed maps: 2) -- for the dense CubeMap the
// two layouts are identical anyway (NUM_TYPES == 3 for both).
template <int _dim, class _Tp> class SparseOrDenseCubeMap {
    static_assert(_dim == 1 || _dim == 2,
                  "SparseOrDenseCubeMap backs edge- or 2-cell-keyed maps");

  public:
    // `cone`: sparse mode only. Virtual cells (cone edges for _dim == 1, cone
    // 2-cells for _dim == 2) get compact ids in a range laid out AFTER the kept
    // cells of that dimension, so the same flat vector backs both.
    SparseOrDenseCubeMap(vector<index_t> shape,
                         const KeptCells *kept = nullptr,
                         const ConeIndex *cone = nullptr)
        : kept(kept), cone(cone) {
        if (kept == nullptr) {
            denseMap.emplace(std::move(shape));
        } else {
            elements.resize(numKept());
        }
    }

    void emplace(uint64_t cube_index, _Tp element) {
        if (kept == nullptr) {
            denseMap->emplace(cube_index, std::move(element));
            return;
        }
        elements[requireCompactIndex(cube_index)] = std::move(element);
    }

    const std::optional<_Tp> &find(uint64_t cube_index) const {
        if (kept == nullptr) {
            return denseMap->find(cube_index);
        }
        if (cube_index != NONE_INDEX) {
            const int32_t compact = compactIndex(cube_index);
            if (compact >= 0) {
                return elements[compact];
            }
        }
        return none;
    }

    optional<_Tp> &operator[](uint64_t cube_index) {
        if (kept == nullptr) {
            return (*denseMap)[cube_index];
        }
        return elements[requireCompactIndex(cube_index)];
    }

    bool hasCone() const { return cone != nullptr; }

    void clear() {
        if (kept == nullptr) {
            denseMap->clear();
        } else {
            elements.clear();
            elements.resize(numKept());
        }
    }

  private:
    size_t numKept() const {
        if constexpr (_dim == 1) {
            return kept->keptEdges.size() +
                   (cone == nullptr ? 0 : cone->numConeEdges());
        } else {
            return kept->keptTwoCells.size() +
                   (cone == nullptr ? 0 : cone->numCone2Cells());
        }
    }
    int32_t compactIndex(uint64_t cube_index) const {
        if (cone != nullptr && isVirtualCell(cube_index)) {
            // A virtual key's anchor is the kept cell it is coned from: a cone
            // edge is anchored at a kept VERTEX, a cone 2-cell at a kept EDGE.
            const index_t x = (cube_index >> 44) & 0xfffff;
            const index_t y = (cube_index >> 24) & 0xfffff;
            const index_t z = (cube_index >> 4) & 0xfffff;
            const uint8_t t = cube_index & 0xf;
            if constexpr (_dim == 1) {
                const int32_t v = kept->vertexCompact(vertexSlot(x, y, z));
                if (v < 0) {
                    return -1;
                }
                return static_cast<int32_t>(kept->keptEdges.size()) +
                       cone->coneEdgeSlot(v, t - VIRTUAL_TYPE_BASE);
            } else {
                const uint8_t anchorType = (t - VIRTUAL_TYPE_BASE) / 4;
                const uint8_t rank = (t - VIRTUAL_TYPE_BASE) % 4;
                const int32_t e = kept->edgeCompact(
                    ((uint64_t)x << 44) | ((uint64_t)y << 24) |
                    ((uint64_t)z << 4) | (uint64_t)anchorType);
                if (e < 0) {
                    return -1;
                }
                return static_cast<int32_t>(kept->keptTwoCells.size()) +
                       cone->cone2CellSlot(e, rank);
            }
        }
        if constexpr (_dim == 1) {
            return kept->edgeCompact(cube_index);
        } else {
            return kept->twoCellCompact(cube_index);
        }
    }
    index_t vertexSlot(index_t x, index_t y, index_t z) const {
        return x * vertexStrideX + y * vertexStrideY + z;
    }
    size_t requireCompactIndex(uint64_t cube_index) const {
        if (cube_index == NONE_INDEX) {
            throw runtime_error("SparseOrDenseCubeMap may not be written at "
                                "NONE_INDEX magic number");
        }
        const int32_t compact = compactIndex(cube_index);
        if (compact < 0) {
            throw runtime_error(
                "sparse invariant violation: sparse CubeMap write to a cell that "
                "is not in the kept complex (dim=" + std::to_string(_dim) +
                ", index=" + std::to_string(cube_index) + ", virtual=" +
                std::to_string((int)isVirtualCell(cube_index)) +
                ", map knows cone cells=" +
                std::to_string((int)(cone != nullptr)) + ")");
        }
        return static_cast<size_t>(compact);
    }

    optional<CubeMap<_dim, _Tp>> denseMap;
    vector<std::optional<_Tp>> elements;
    const KeptCells *kept;
    const ConeIndex *cone = nullptr;
    size_t vertexStrideX = 0;
    size_t vertexStrideY = 0;
    std::optional<_Tp> none = {};

  public:
    // The dense vertex id space (x*n_yz + y*n_z + z) is needed to resolve a
    // cone edge's anchor vertex; it is set once from the complex's shape.
    void setVertexStrides(size_t strideX, size_t strideY) {
        vertexStrideX = strideX;
        vertexStrideY = strideY;
    }
};
} // namespace dim3
