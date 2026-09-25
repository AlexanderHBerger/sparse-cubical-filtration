#pragma once

#include "../config.h"

#include <cstdint>
#include <vector>

using namespace std;

namespace dim2 {
typedef tuple<index_t, index_t> Coordinate;
typedef vector<Coordinate> RepresentativeCycle;

class Cube {
  public:
    value_t birth;
    uint64_t index;

    Cube();
    Cube(value_t birth, index_t x, index_t y, uint8_t type);
    Cube(const Cube &cube);
    index_t x() const;
    index_t y() const;
    uint8_t type() const;
    bool operator==(const Cube &rhs) const;
    void print() const;
};

struct CubeComparator {
    bool operator()(const Cube &Cube1, const Cube &Cube2) const;
};

class Pair {
  public:
    const Cube birth;
    const Cube death;

    Pair();
    Pair(const Cube &birth, const Cube &death);
    Pair(const Pair &pair);
    bool operator==(const Pair &rhs) const;
    void print() const;
};

class Match {
  public:
    Pair pair0;
    Pair pair1;

    Match(Pair pair0, Pair pair1);
    void print() const;
};

class CubicalGridComplex {
  public:
    CubicalGridComplex(const vector<value_t> &image,
                       const vector<index_t> &shape);
    CubicalGridComplex(CubicalGridComplex &&other);
    ~CubicalGridComplex();
    size_t getNumberOfCubes(const uint8_t &dim) const;
    value_t getBirth(const index_t &x, const index_t &y) const;
    value_t getBirth(const index_t &x, const index_t &y, const uint8_t &type,
                     const uint8_t &dim) const;
    dim2::Coordinate getParentVoxel(const Cube &c, const uint8_t &dim) const;
    void printImage() const;
    void
    printRepresentativeCycle(const dim2::RepresentativeCycle &reprCycle) const;
    const vector<index_t> shape;
    const index_t m_x;
    const index_t m_y;
    const index_t m_xy;
    const index_t n_xy;

  private:
    void getGridFromVector(const vector<value_t> &vector);
    // Contiguous padded (shape+2)^2 grid, row-major (see src_3D).
    vector<value_t> grid;
    size_t gridStrideX;
};

// Kept-cell reindexing, 2D variant of the src_3D
// KeptCells: one O(N) pass over the comparison complex classifies every
// cell as kept (finite birth) or masked. Masked voxels are INFTY in BOTH
// inputs (shared mask), so a cell has finite birth in the comparison
// complex iff it does in input 0 iff in input 1 -- this single
// classification serves all three complexes and both image complexes.
// It provides
//   - dense int32 compact-id arrays over the SAME slot spaces the dense
//     structures use (vertices: the dim-0 UnionFind vertex ids; top cells:
//     the UnionFindDual node ids), -1 marking masked/absent cells, and
//   - the kept-cell lists in the SAME canonical nested-loop order
//     (x,y[,type]) as the full-grid enumerations, so downstream sorts and
//     processing orders see identical sequences.
// A kept cell may have a finite birth >= maskThreshold in one input (it is
// kept because the other input is below threshold there). Unlike 3D there are no
// CubeMaps in 2D, so no compact-id array over the edge slot space exists --
// only the kept edge list (packed cube keys x<<34 | y<<4 | type), which
// backs the dual-edge enumerations.
class KeptCells {
  public:
    explicit KeptCells(const CubicalGridComplex &cgcComp);

    int32_t vertexCompact(index_t vertexIdx) const {
        return compactVertex[vertexIdx];
    }
    int32_t topCellCompact(index_t dualNodeIdx) const {
        return compactTopCell[dualNodeIdx];
    }

    // Kept lists in canonical order: dense vertex ids (x*n_y + y), packed
    // cube spatial keys (x<<34 | y<<4 | type), and dual node ids (x*m_y + y)
    // respectively.
    vector<index_t> keptVertices;
    vector<uint64_t> keptEdges;
    vector<index_t> keptTopCells;

  private:
    vector<int32_t> compactVertex;
    vector<int32_t> compactTopCell;
};

// Sparse mode: the virtual nodes that stand in for the masked region, which
// enters as one terminal slice at CONE_BIRTH. Derived in one pass from the
// caller-supplied voxel-level pocket labels.
//
// A top cell is non-kept iff at least one of its 2**ndim corners is masked, and
// all of its masked corners lie within L_inf <= 1 of each other, hence in the
// same fine pocket -- so "the pocket of a non-kept top cell" is well defined and
// is a max over the corner labels.
class PocketNodes {
  public:
    PocketNodes(const CubicalGridComplex &cgcComp,
                const vector<int32_t> &voxelLabels);

    int32_t numPockets() const { return n; }
    // Dual node id (x*m_y + y) -> pocket id in [1, n], or 0 if the top cell is
    // kept. Never out of range: the array covers every dual node.
    int32_t ofTopCell(index_t nodeIdx) const { return topCellPocket[nodeIdx]; }
    // A pocket reaches the exterior iff it contains a masked voxel on a
    // grid face: such a voxel makes some boundary 2-cell non-kept, and the top
    // cell adjacent to that 2-cell belongs to this pocket. Those pockets ARE
    // the exterior and are pre-unioned with the sentinel, so they never die.
    bool touchesBoundary(int32_t pocket) const { return boundary[pocket] != 0; }

  private:
    int32_t n = 0;
    vector<int32_t> topCellPocket;
    vector<uint8_t> boundary;
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
    dim2::Coordinate getCoordinates(index_t idx) const;
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
    // node (compact id size()-1, original id m_xy); equal-birth tie-breaks
    // compare original dense node ids.
    // Sparse mode adds one node per pocket AFTER the exterior sentinel,
    // all born at `pocketBirth` (= CONE_BIRTH, the value of every masked
    // cell). Node layout is then
    //     [kept top cells 0 .. K-1] [exterior sentinel K] [pockets K+1 .. K+P]
    // so `K + pocketId` addresses pocket `pocketId` and `K + 0` is the
    // sentinel -- which makes the out-of-grid case and the masked case one
    // uniform lookup in getBoundaryIndices.
    UnionFindDual(const CubicalGridComplex &cgc,
                  const KeptCells *kept = nullptr,
                  const PocketNodes *pockets = nullptr,
                  value_t pocketBirth = INFTY);
    // True iff `idx` is one of the virtual pocket nodes (never the sentinel).
    // Such a node has no voxel: a bar whose dying root is virtual is CENSORED
    // (its death is CONE_BIRTH) and must be reported by its birth cell only.
    bool isPocketNode(index_t idx) const {
        return pockets != nullptr && idx > sentinelIdx;
    }
    index_t sentinel() const { return sentinelIdx; }
    // Pre-union every boundary-touching pocket into the exterior sentinel.
    // Must be called after each reset(), before the sweep.
    void seedPockets();
    index_t find(index_t x);
    index_t link(index_t x, index_t y);
    value_t getBirth(const index_t &idx) const;
    dim2::Coordinate getCoordinates(index_t idx) const;
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
} // namespace dim2
