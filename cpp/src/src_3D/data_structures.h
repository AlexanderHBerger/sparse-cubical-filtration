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
    vector<value_t> grid;
    size_t gridStrideX;
    size_t gridStrideY;
};

// Cells with finite birth in the comparison complex. Kept lists follow the
// order of the full-grid enumerations, compact ids index the sparse maps.
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

    vector<index_t> keptVertices;
    vector<uint64_t> keptEdges;
    vector<uint64_t> keptTwoCells;
    vector<index_t> keptTopCells;

  private:
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

class PocketNodes {
  public:
    PocketNodes(const CubicalGridComplex &cgcComp,
                const vector<int32_t> &voxelLabels, size_t keptVoxels);

    int32_t numPockets() const { return n; }
    // All masked corners of a top cell lie in the same pocket, so its pocket
    // is the max over the corner labels (0 if the cell is kept).
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
    int32_t ofTopCell(index_t nodeIdx) const {
        return ofTopCellAt(nodeIdx / myz, nodeIdx / mz % my, nodeIdx % mz);
    }

    bool touchesBoundary(int32_t pocket) const { return boundary[pocket] != 0; }

  private:
    const vector<int32_t> *labels = nullptr;
    vector<int32_t> cubePocket;
    size_t syz = 0;
    size_t sz = 0;
    index_t mx = 0, my = 0, mz = 0;
    size_t myz = 0;
    int32_t n = 0;
    vector<uint8_t> boundary;
};

// Virtual cells use the unused values 3..14 of the 4-bit type field and are
// anchored at the kept cell they are coned from:
//   coneEdge(p, v):  type = 3 + rank
//   cone2Cell(p, e): type = 3 + 4 * e.type + rank
// where rank is the position of pocket p among the sorted pockets touching
// the anchor. At equal coordinates they sort after the real cells.
constexpr uint8_t VIRTUAL_TYPE_BASE = 3;
inline bool isVirtualCell(uint64_t index) {
    return (index & 0xf) >= VIRTUAL_TYPE_BASE;
}

// Cone over the rim of every pocket (dimension 1 in 3D). A kept vertex or edge
// is on the rim of pocket p if it is a face of a non-kept cube of p. Every
// (rim vertex, p) gets a cone edge and every (rim edge, p) a cone 2-cell.
class ConeIndex {
  public:
    ConeIndex(const CubicalGridComplex &cgcComp, const KeptCells &kept,
              const PocketNodes &pockets, bool prune = false);

    size_t numConeEdges() const { return vPocket.size(); }
    size_t numCone2Cells() const { return ePocket.size(); }

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
    uint8_t rankAtV0(int32_t keptEdge, uint8_t rank) const {
        return eRankV0[eStart[keptEdge] + rank];
    }
    uint8_t rankAtV1(int32_t keptEdge, uint8_t rank) const {
        return eRankV1[eStart[keptEdge] + rank];
    }

    bool emitCone2Cell(int32_t slot) const {
        return (emitBits[static_cast<size_t>(slot) >> 6] >>
                (static_cast<size_t>(slot) & 63)) &
               1ULL;
    }
    size_t numEmittedCone2Cells() const { return nEmitted; }
    size_t numRimFaces() const { return fPocket.size(); }

    // With pruning, the cone block is a spanning forest of the rim graph plus
    // one column per non-tree slot. Tree columns pair with a cone edge at
    // their own birth and produce no bar, so only the non-tree columns are
    // emitted, each as its fundamental cycle e + path(u, v) over kept edges.
    bool hasCycles() const { return !cycPtr.empty(); }
    bool emitCycleColumn(int32_t slot) const {
        return cycIndex[static_cast<size_t>(slot)] >= 0;
    }
    int32_t cycleLength(int32_t slot) const {
        const int32_t c = cycIndex[static_cast<size_t>(slot)];
        return cycPtr[c + 1] - cycPtr[c];
    }
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
    vector<int32_t> fStart, fPocket;
    vector<int32_t> efStart, efFace;
    vector<uint64_t> emitBits;
    size_t nEmitted = 0;
    vector<int32_t> cycIndex;
    vector<int32_t> cycPtr;
    vector<uint64_t> cycEdge;

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
    vector<index_t> original;
    const CubicalGridComplex &cgc;
    const KeptCells *kept;
};

// In sparse mode the nodes are laid out as
//   [kept top cells 0..K-1] [exterior sentinel K] [pockets K+1..K+P].
class UnionFindDual {
  public:
    UnionFindDual(const CubicalGridComplex &cgc,
                  const KeptCells *kept = nullptr,
                  const PocketNodes *pockets = nullptr,
                  value_t pocketBirth = INFTY);
    bool isPocketNode(index_t idx) const {
        return pockets != nullptr && idx > sentinelIdx;
    }
    index_t sentinel() const { return sentinelIdx; }
    // Has to be called again after every reset().
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

template <int _dim, class _Tp> class SparseOrDenseCubeMap {
    static_assert(_dim == 1 || _dim == 2,
                  "SparseOrDenseCubeMap backs edge- or 2-cell-keyed maps");

  public:
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
    void setVertexStrides(size_t strideX, size_t strideY) {
        vertexStrideX = strideX;
        vertexStrideY = strideY;
    }
};
} // namespace dim3
