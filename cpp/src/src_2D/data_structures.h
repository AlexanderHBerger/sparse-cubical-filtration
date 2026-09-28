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
    vector<value_t> grid;
    size_t gridStrideX;
};

// Cells with finite birth in the comparison complex, in the order of the
// full-grid enumerations (see src_3D).
class KeptCells {
  public:
    explicit KeptCells(const CubicalGridComplex &cgcComp);

    int32_t vertexCompact(index_t vertexIdx) const {
        return compactVertex[vertexIdx];
    }
    int32_t topCellCompact(index_t dualNodeIdx) const {
        return compactTopCell[dualNodeIdx];
    }

    vector<index_t> keptVertices;
    vector<uint64_t> keptEdges;
    vector<index_t> keptTopCells;

  private:
    vector<int32_t> compactVertex;
    vector<int32_t> compactTopCell;
};

class PocketNodes {
  public:
    PocketNodes(const CubicalGridComplex &cgcComp,
                const vector<int32_t> &voxelLabels);

    int32_t numPockets() const { return n; }
    int32_t ofTopCell(index_t nodeIdx) const { return topCellPocket[nodeIdx]; }
    bool touchesBoundary(int32_t pocket) const { return boundary[pocket] != 0; }

  private:
    int32_t n = 0;
    vector<int32_t> topCellPocket;
    vector<uint8_t> boundary;
};

class UnionFind {
  public:
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
    dim2::Coordinate getCoordinates(index_t idx) const;
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
} // namespace dim2
