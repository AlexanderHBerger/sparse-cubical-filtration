#include "data_structures.h"
#include <algorithm>
#include <iostream>

using namespace dim2;
using namespace std;

Cube::Cube() : birth(0), index(NONE_INDEX) {}

Cube::Cube(const Cube &cube) : birth(cube.birth), index(cube.index) {}

Cube::Cube(value_t _birth, index_t _x, index_t _y, uint8_t _type)
    : birth(_birth) {
    index = ((uint64_t)_x << 34) | ((uint64_t)_y << 4) | (uint64_t)_type;
}

index_t Cube::x() const { return ((index >> 34) & 0xfffff); }

index_t Cube::y() const { return ((index >> 4) & 0xfffff); }

uint8_t Cube::type() const { return (index & 0xf); }

bool Cube::operator==(const Cube &rhs) const { return (index == rhs.index); }

void Cube::print() const {
    cout << "(" << birth << "," << x() << "," << y() << "," << unsigned(type())
         << ")";
}

bool CubeComparator::operator()(const Cube &cube1, const Cube &cube2) const {
    if (cube1.birth == cube2.birth) {
        return (cube1.index < cube2.index);
    } else {
        return (cube1.birth < cube2.birth);
    }
}

Pair::Pair() {}

Pair::Pair(const Cube &_birth, const Cube &_death)
    : birth(_birth), death(_death) {}

Pair::Pair(const Pair &pair) : birth(pair.birth), death(pair.death) {}

bool Pair::operator==(const Pair &rhs) const {
    return (birth == rhs.birth && death == rhs.death);
}

void Pair::print() const {
    cout << "(";
    birth.print();
    cout << ";";
    death.print();
    cout << ")";
}

Match::Match(Pair _pair0, Pair _pair1) : pair0(_pair0), pair1(_pair1) {}

void Match::print() const {
    pair0.print();
    cout << " <-> ";
    pair1.print();
    cout << endl;
}

CubicalGridComplex::CubicalGridComplex(const vector<value_t> &image,
                                       const vector<index_t> &_shape)
    : shape(_shape), m_x(shape[0] - 1), m_y(shape[1] - 1), m_xy(m_x * m_y),
      n_xy(shape[0] * shape[1]),
      gridStrideX(static_cast<size_t>(shape[1] + 2)) {
    getGridFromVector(image);
}

CubicalGridComplex::CubicalGridComplex(CubicalGridComplex &&other)
    : shape(std::move(other.shape)), m_x(other.m_x), m_y(other.m_y),
      m_xy(other.m_xy), n_xy(other.n_xy), grid(std::move(other.grid)),
      gridStrideX(other.gridStrideX) {}

CubicalGridComplex::~CubicalGridComplex() {}

size_t CubicalGridComplex::getNumberOfCubes(const uint8_t &dim) const {
    switch (dim) {
    case 0:
        return n_xy;

    case 1:
        return m_x * shape[1] + shape[0] * m_y;

    case 2:
        return m_xy;
    }
    throw runtime_error("No cubes in dim " + std::to_string(unsigned(dim)));
}

value_t CubicalGridComplex::getBirth(const index_t &x, const index_t &y) const {
    return grid[static_cast<size_t>(x + 1) * gridStrideX + (y + 1)];
}

value_t CubicalGridComplex::getBirth(const index_t &x, const index_t &y,
                                     const uint8_t &type,
                                     const uint8_t &dim) const {
    switch (dim) {
    case 0:
        return getBirth(x, y);

    case 1:
        switch (type) {
        case 0:
            return max(getBirth(x, y), getBirth(x + 1, y));

        case 1:
            return max(getBirth(x, y), getBirth(x, y + 1));
        }

    case 2:
        return max({getBirth(x, y), getBirth(x, y + 1), getBirth(x + 1, y),
                    getBirth(x + 1, y + 1)});
    }
    throw runtime_error("Birth not found!");
}

dim2::Coordinate CubicalGridComplex::getParentVoxel(const Cube &cube,
                                                    const uint8_t &dim) const {
    index_t x = cube.x();
    index_t y = cube.y();
    switch (dim) {
    case 0:
        return {x, y};

    case 1:
        switch (cube.type()) {
        case 0:
            if (cube.birth == getBirth(x + 1, y)) {
                return {x + 1, y};
            } else {
                return {x, y};
            }

        case 1:
            if (cube.birth == getBirth(x, y + 1)) {
                return {x, y + 1};
            } else {
                return {x, y};
            }
        }

    case 2:
        if (cube.birth == getBirth(x + 1, y + 1)) {
            return {x + 1, y + 1};
        } else if (cube.birth == getBirth(x + 1, y)) {
            return {x + 1, y};
        } else if (cube.birth == getBirth(x, y + 1)) {
            return {x, y + 1};
        } else {
            return {x, y};
        }
    }
    throw runtime_error("Parent voxel not found!");
}

void CubicalGridComplex::printImage() const {
    value_t birth;
    for (index_t x = 0; x < shape[0]; ++x) {
        for (index_t y = 0; y < shape[1]; ++y) {
            birth = getBirth(x, y);
            if (birth < 10) {
                cout << ' ' << birth << ' ';
            } else {
                cout << birth << ' ';
            }
        }
        cout << endl;
    }
    cout << endl;
}

void CubicalGridComplex::printRepresentativeCycle(
    const dim2::RepresentativeCycle &reprCycle) const {
    for (index_t x = 0; x < shape[0]; ++x) {
        for (index_t y = 0; y < shape[1]; ++y) {
            auto it = find(reprCycle.begin(), reprCycle.end(),
                           dim2::Coordinate{x, y});
            if (it == reprCycle.begin()) {
                cout << "2  ";
            } else if (it == reprCycle.end() - 1) {
                cout << "-1 ";
            } else if (it != reprCycle.end()) {
                cout << "1  ";
            } else {
                cout << "0  ";
            }
        }
        cout << endl;
    }
}

void CubicalGridComplex::getGridFromVector(const vector<value_t> &vec) {
    const size_t px = static_cast<size_t>(shape[0]) + 2;
    const size_t py = static_cast<size_t>(shape[1]) + 2;
    grid.assign(px * py, INFTY);
    size_t counter = 0;
    for (index_t x = 1; x <= shape[0]; ++x) {
        value_t *row = grid.data() + x * gridStrideX;
        for (index_t y = 1; y <= shape[1]; ++y) {
            row[y] = vec[counter++];
        }
    }
}

KeptCells::KeptCells(const CubicalGridComplex &cgcComp) {
    const vector<index_t> &shape = cgcComp.shape;

    const size_t numVertices = static_cast<size_t>(shape[0]) * shape[1];
    compactVertex.assign(numVertices, -1);
    {
        size_t slot = 0;
        for (index_t x = 0; x < shape[0]; ++x) {
            for (index_t y = 0; y < shape[1]; ++y, ++slot) {
                if (cgcComp.getBirth(x, y) != INFTY) {
                    compactVertex[slot] =
                        static_cast<int32_t>(keptVertices.size());
                    keptVertices.push_back(static_cast<index_t>(slot));
                }
            }
        }
    }

    for (index_t x = 0; x < shape[0]; ++x) {
        for (index_t y = 0; y < shape[1]; ++y) {
            for (uint8_t type = 0; type < 2; ++type) {
                if (cgcComp.getBirth(x, y, type, 1) != INFTY) {
                    keptEdges.push_back(((uint64_t)x << 34) |
                                        ((uint64_t)y << 4) | (uint64_t)type);
                }
            }
        }
    }

    const size_t numTopCells = cgcComp.getNumberOfCubes(2);
    compactTopCell.assign(numTopCells, -1);
    {
        size_t node = 0;
        for (index_t x = 0; x < cgcComp.m_x; ++x) {
            for (index_t y = 0; y < cgcComp.m_y; ++y, ++node) {
                if (cgcComp.getBirth(x, y, 0, 2) != INFTY) {
                    compactTopCell[node] =
                        static_cast<int32_t>(keptTopCells.size());
                    keptTopCells.push_back(static_cast<index_t>(node));
                }
            }
        }
    }
}

PocketNodes::PocketNodes(const CubicalGridComplex &cgcComp,
                         const vector<int32_t> &voxelLabels) {
    const index_t sx = cgcComp.shape[0];
    const index_t sy = cgcComp.shape[1];
    if (voxelLabels.size() != static_cast<size_t>(sx) * static_cast<size_t>(sy)) {
        throw runtime_error("pocket_labels size does not match the input shape");
    }
    for (const int32_t v : voxelLabels) {
        if (v > n) {
            n = v;
        }
        if (v < 0) {
            throw runtime_error("pocket_labels must be non-negative");
        }
    }
    for (index_t x = 0; x < sx; ++x) {
        for (index_t y = 0; y < sy; ++y) {
            const bool masked = cgcComp.getBirth(x, y) == INFTY;
            if ((voxelLabels[x * sy + y] != 0) != masked) {
                throw runtime_error(
                    "pocket_labels support does not match the mask: labels "
                    "must be nonzero exactly on the masked (INFTY) voxels");
            }
        }
    }
    boundary.assign(static_cast<size_t>(n) + 1, 0);
    topCellPocket.assign(static_cast<size_t>(cgcComp.m_xy), 0);
    for (index_t x = 0; x < cgcComp.m_x; ++x) {
        for (index_t y = 0; y < cgcComp.m_y; ++y) {
            int32_t best = 0;
            for (index_t dx = 0; dx < 2; ++dx) {
                for (index_t dy = 0; dy < 2; ++dy) {
                    const int32_t lab =
                        voxelLabels[(x + dx) * sy + (y + dy)];
                    if (lab > best) {
                        best = lab;
                    }
                }
            }
            topCellPocket[x * cgcComp.m_y + y] = best;
        }
    }
    for (index_t x = 0; x < sx; ++x) {
        for (index_t y = 0; y < sy; ++y) {
            if (x != 0 && x != sx - 1 && y != 0 && y != sy - 1) {
                continue;
            }
            const int32_t lab = voxelLabels[x * sy + y];
            if (lab > 0) {
                boundary[lab] = 1;
            }
        }
    }
}

UnionFind::UnionFind(const CubicalGridComplex &_cgc, const KeptCells *_kept)
    : cgc(_cgc), kept(_kept) {
    if (kept == nullptr) {
        size_t n = cgc.getNumberOfCubes(0);
        parent.reserve(n);
        birthtime.reserve(n);
        index_t counter = 0;
        for (index_t x = 0; x < _cgc.shape[0]; ++x) {
            for (index_t y = 0; y < _cgc.shape[1]; ++y) {
                parent.push_back(counter++);
                birthtime.push_back(cgc.getBirth(x, y));
            }
        }
        return;
    }
    const size_t n = kept->keptVertices.size();
    parent.reserve(n);
    birthtime.reserve(n);
    original = kept->keptVertices;
    index_t counter = 0;
    for (const index_t vertexIdx : kept->keptVertices) {
        parent.push_back(counter++);
        birthtime.push_back(
            cgc.getBirth(vertexIdx / cgc.shape[1], vertexIdx % cgc.shape[1]));
    }
}

index_t UnionFind::find(index_t x) {
    index_t y = x, z = parent[y];
    while (z != y) {
        y = z;
        z = parent[y];
    }
    y = parent[x];
    while (z != y) {
        parent[x] = z;
        x = y;
        y = parent[x];
    }
    return z;
}

index_t UnionFind::link(index_t x, index_t y) {
    if (birthtime[x] > birthtime[y]) {
        parent[x] = y;
        return x;
    } else if (birthtime[x] < birthtime[y]) {
        parent[y] = x;
        return y;
    } else {
        const bool xDies =
            (kept == nullptr) ? (x > y) : (original[x] > original[y]);
        if (xDies) {
            parent[x] = y;
            return x;
        } else {
            parent[y] = x;
            return y;
        }
    }
}

value_t UnionFind::getBirth(const index_t &idx) const { return birthtime[idx]; }

dim2::Coordinate UnionFind::getCoordinates(index_t idx) const {
    if (kept != nullptr) {
        idx = original[idx];
    }
    return {idx / cgc.shape[1], idx % cgc.shape[1]};
}

vector<index_t> UnionFind::getBoundaryIndices(const Cube &edge) const {
    vector<index_t> boundaryIndices(2);
    switch (edge.type()) {
    case 0:
        boundaryIndices[0] = edge.x() * cgc.shape[1] + edge.y();
        boundaryIndices[1] = (edge.x() + 1) * cgc.shape[1] + edge.y();
        break;

    case 1:
        boundaryIndices[0] = edge.x() * cgc.shape[1] + edge.y();
        boundaryIndices[1] = edge.x() * cgc.shape[1] + edge.y() + 1;
        break;
    }
    if (kept != nullptr) {
        for (int i = 0; i < 2; ++i) {
            const int32_t compact = kept->vertexCompact(boundaryIndices[i]);
            if (compact < 0) {
                throw runtime_error("sparse invariant violation: kept edge with "
                                    "a masked endpoint vertex");
            }
            boundaryIndices[i] = static_cast<index_t>(compact);
        }
    }
    return boundaryIndices;
}

void UnionFind::reset() {
    for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = i;
    }
}

UnionFindDual::UnionFindDual(const CubicalGridComplex &_cgc,
                             const KeptCells *_kept,
                             const PocketNodes *_pockets, value_t pocketBirth)
    : cgc(_cgc), kept(_kept), pockets(_pockets) {
    if (kept != nullptr && pockets == nullptr) {
        throw runtime_error(
            "sparse UnionFindDual requires the pocket nodes");
    }
    if (kept == nullptr) {
        index_t n = cgc.getNumberOfCubes(2) + 1;
        parent.reserve(n);
        birthtime.reserve(n);
        index_t counter = 0;
        for (index_t x = 0; x < _cgc.m_x; ++x) {
            for (index_t y = 0; y < _cgc.m_y; ++y) {
                parent.push_back(counter++);
                birthtime.push_back(cgc.getBirth(x, y, 0, 2));
            }
        }
        parent.push_back(counter);
        birthtime.push_back(INFTY);
        sentinelIdx = static_cast<index_t>(parent.size() - 1);
        return;
    }
    const size_t n = kept->keptTopCells.size() + 1;
    parent.reserve(n);
    birthtime.reserve(n);
    original.reserve(n);
    index_t counter = 0;
    for (const index_t nodeIdx : kept->keptTopCells) {
        parent.push_back(counter++);
        birthtime.push_back(
            cgc.getBirth(nodeIdx / cgc.m_y, nodeIdx % cgc.m_y, 0, 2));
        original.push_back(nodeIdx);
    }
    parent.push_back(counter);
    birthtime.push_back(INFTY);
    original.push_back(cgc.m_xy);
    sentinelIdx = static_cast<index_t>(parent.size() - 1);
    if (pockets != nullptr) {
        for (int32_t p = 1; p <= pockets->numPockets(); ++p) {
            parent.push_back(static_cast<index_t>(parent.size()));
            birthtime.push_back(pocketBirth);
            original.push_back(cgc.m_xy + p);
        }
    }
}

void UnionFindDual::seedPockets() {
    if (pockets == nullptr) {
        return;
    }
    // Pockets that touch the grid boundary are part of the exterior.
    for (int32_t p = 1; p <= pockets->numPockets(); ++p) {
        if (pockets->touchesBoundary(p)) {
            parent[sentinelIdx + p] = sentinelIdx;
        }
    }
}

index_t UnionFindDual::find(index_t x) {
    index_t y = x, z = parent[y];
    while (z != y) {
        y = z;
        z = parent[y];
    }
    y = parent[x];
    while (z != y) {
        parent[x] = z;
        x = y;
        y = parent[x];
    }
    return z;
}

index_t UnionFindDual::link(index_t x, index_t y) {
    if (birthtime[x] < birthtime[y]) {
        parent[x] = y;
        return x;
    } else if (birthtime[x] > birthtime[y]) {
        parent[y] = x;
        return y;
    } else {
        const bool xDies =
            (kept == nullptr) ? (x < y) : (original[x] < original[y]);
        if (xDies) {
            parent[x] = y;
            return x;
        } else {
            parent[y] = x;
            return y;
        }
    }
}

value_t UnionFindDual::getBirth(const index_t &idx) const {
    return birthtime[idx];
}

dim2::Coordinate UnionFindDual::getCoordinates(index_t idx) const {
    if (isPocketNode(idx) || idx == sentinelIdx) {
        throw runtime_error(
            "UnionFindDual::getCoordinates on a virtual node (pocket or "
            "exterior sentinel): censored bars must be reported by their "
            "birth cell");
    }
    if (kept != nullptr) {
        idx = original[idx];
    }
    return {idx / cgc.m_y, idx % cgc.m_y};
}

vector<index_t> UnionFindDual::getBoundaryIndices(const Cube &edge) const {
    vector<index_t> boundaryIndices(2);
    switch (edge.type()) {
    case 0:
        if (edge.y() == 0) {
            boundaryIndices[0] = cgc.m_xy;
        } else {
            boundaryIndices[0] = edge.x() * cgc.m_y + edge.y() - 1;
        }
        if (edge.y() == cgc.m_y) {
            boundaryIndices[1] = cgc.m_xy;
        } else {
            boundaryIndices[1] = edge.x() * cgc.m_y + edge.y();
        }
        break;

    case 1:
        if (edge.x() == 0) {
            boundaryIndices[0] = cgc.m_xy;
        } else {
            boundaryIndices[0] = (edge.x() - 1) * cgc.m_y + edge.y();
        }
        if (edge.x() == cgc.m_x) {
            boundaryIndices[1] = cgc.m_xy;
        } else {
            boundaryIndices[1] = edge.x() * cgc.m_y + edge.y();
        }
        break;
    }
    if (kept != nullptr) {
        for (int i = 0; i < 2; ++i) {
            if (boundaryIndices[i] == static_cast<index_t>(cgc.m_xy)) {
                boundaryIndices[i] = sentinelIdx;
            } else {
                const int32_t compact =
                    kept->topCellCompact(boundaryIndices[i]);
                if (compact >= 0) {
                    boundaryIndices[i] = static_cast<index_t>(compact);
                } else {
                    boundaryIndices[i] =
                        sentinelIdx +
                        static_cast<index_t>(pockets->ofTopCell(
                            boundaryIndices[i]));
                }
            }
        }
    }
    return boundaryIndices;
}

void UnionFindDual::reset() {
    for (size_t i = 0; i < parent.size(); ++i) {
        parent[i] = i;
    }
}
