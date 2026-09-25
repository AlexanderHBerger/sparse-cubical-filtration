#include "data_structures.h"
#include "BettiMatching.h"
#include <algorithm>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <chrono>

using namespace dim3;
using namespace std;

#ifdef RUNTIME
namespace dim3 {
double g_tGrid = 0, g_tKept = 0, g_tPocketVal = 0, g_tPocketTop = 0,
       g_tConeVE = 0, g_tConeFaces = 0, g_tConeInc = 0,
       g_tConeEmit = 0, g_tConeCycles = 0, g_tRangeCheck = 0;
}
#define RT_NOW() std::chrono::high_resolution_clock::now()
#define RT_SINCE(t) (std::chrono::duration<double>(RT_NOW() - (t)).count())
#endif

Cube::Cube() : birth(0), index(NONE_INDEX) {}

Cube::Cube(value_t _birth, index_t _x, index_t _y, index_t _z, uint8_t _type)
    : birth(_birth) {
    index = ((uint64_t)_x << 44) | ((uint64_t)_y << 24) | ((uint64_t)_z << 4) |
            (uint64_t)_type;
}

Cube::Cube(value_t _birth, vector<index_t> coordinates, uint8_t _type)
    : birth(_birth) {
    index = ((uint64_t)coordinates[0] << 44) |
            ((uint64_t)coordinates[1] << 24) | ((uint64_t)coordinates[2] << 4) |
            (uint64_t)_type;
}

Cube::Cube(const Cube &cube) : birth(cube.birth), index(cube.index) {}

bool Cube::operator==(const Cube &rhs) const { return (index == rhs.index); }

index_t Cube::x() const { return ((index >> 44) & 0xfffff); }

index_t Cube::y() const { return ((index >> 24) & 0xfffff); }

index_t Cube::z() const { return ((index >> 4) & 0xfffff); }

uint8_t Cube::type() const { return (index & 0xf); }

void Cube::print() const {
    cout << "(" << birth << "," << x() << "," << y() << "," << z() << ","
         << unsigned(type()) << ")";
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
    : shape(_shape), m_x(shape[0] - 1), m_y(shape[1] - 1), m_z(shape[2] - 1),
      m_yz(m_y * m_z), m_xyz(m_x * m_yz), n_yz(shape[1] * shape[2]),
      n_xyz(shape[0] * n_yz),
      gridStrideX(static_cast<size_t>(shape[1] + 2) * (shape[2] + 2)),
      gridStrideY(static_cast<size_t>(shape[2] + 2)) {
    getGridFromVector(image);
}

CubicalGridComplex::CubicalGridComplex(CubicalGridComplex &&other)
    : shape(std::move(other.shape)), m_x(other.m_x), m_y(other.m_y),
      m_z(other.m_z), m_yz(other.m_yz), m_xyz(other.m_xyz),
      n_yz(other.n_yz), n_xyz(other.n_xyz), grid(std::move(other.grid)),
      gridStrideX(other.gridStrideX), gridStrideY(other.gridStrideY) {}

CubicalGridComplex::~CubicalGridComplex() {}

size_t CubicalGridComplex::getNumberOfCubes(const uint8_t &dim) const {
    switch (dim) {
    case 0:
        return shape[0] * shape[1] * shape[2];

    case 1:
        return m_x * shape[1] * shape[2] + shape[0] * m_y * shape[2] +
               shape[0] * shape[1] * m_z;

    case 2:
        return m_x * m_y * shape[2] + m_x * shape[1] * m_z + shape[0] * m_yz;

    case 3:
        return m_x * m_y * m_z;
    }
    throw runtime_error("No cubes in dim " + std::to_string(unsigned(dim)));
}

value_t CubicalGridComplex::getBirth(const index_t &x, const index_t &y,
                                     const index_t &z) const {
    return grid[static_cast<size_t>(x + 1) * gridStrideX +
                static_cast<size_t>(y + 1) * gridStrideY + (z + 1)];
}

value_t CubicalGridComplex::getBirth(const index_t &x, const index_t &y,
                                     const index_t &z, const uint8_t &type,
                                     const uint8_t &dim) const {
    // The inner switches below have no default and no break, so a type outside
    // 0..2 falls through case 1 -> case 2 -> case 3 and silently returns the
    // CUBE's 8-corner birth. Virtual (cone) cells carry type 3..14, so without
    // this guard they read a neighbouring cube's value -- usually INFTY, since
    // cone cells hug masked regions -- instead of failing.
    if (dim <= 2 && type > 2) {
        throw runtime_error(
            "getBirth on a virtual (cone) cell: its birth is not a grid value "
            "but CONE_BIRTH, and must be supplied by the caller");
    }
    switch (dim) {
    case 0:
        return getBirth(x, y, z);

    case 1:
        switch (type) {
        case 0:
            return max(getBirth(x, y, z), getBirth(x + 1, y, z));

        case 1:
            return max(getBirth(x, y, z), getBirth(x, y + 1, z));

        case 2:
            return max(getBirth(x, y, z), getBirth(x, y, z + 1));
        }

    case 2:
        switch (type) {
        case 0:
            return max({getBirth(x, y, z), getBirth(x, y, z + 1),
                        getBirth(x, y + 1, z), getBirth(x, y + 1, z + 1)});

        case 1:
            return max({getBirth(x, y, z), getBirth(x, y, z + 1),
                        getBirth(x + 1, y, z), getBirth(x + 1, y, z + 1)});

        case 2:
            return max({getBirth(x, y, z), getBirth(x, y + 1, z),
                        getBirth(x + 1, y, z), getBirth(x + 1, y + 1, z)});
        }

    case 3:
        return max({getBirth(x, y, z), getBirth(x, y, z + 1),
                    getBirth(x, y + 1, z), getBirth(x, y + 1, z + 1),
                    getBirth(x + 1, y, z), getBirth(x + 1, y, z + 1),
                    getBirth(x + 1, y + 1, z), getBirth(x + 1, y + 1, z + 1)});
    }
    throw runtime_error("Birth not found!");
}

Coordinate CubicalGridComplex::getParentVoxel(const Cube &cube,
                                              const uint8_t &dim) const {
    // Virtual (cone) cells have no voxel at all, and the switch below does
    // NOT reject them on its own: the inner `switch (cube.type())` under
    // `case 2` handles only types 0..2, so a virtual type falls THROUGH into
    // `case 3` -- the 3-cell corner search -- which probes (x+1, y+1, z+1) and
    // returns an out-of-grid coordinate at a boundary and a silently wrong
    // in-grid voxel everywhere else. Every caller must therefore route virtual
    // cells elsewhere before reaching here; make a miss loud rather than
    // silently corrupt. Never fires in dense mode -- no cell is virtual
    // there.
    if (isVirtualCell(cube.index)) {
        throw runtime_error(
            "getParentVoxel called on a virtual (cone) cell, which has no "
            "voxel; the caller must handle censored deaths explicitly");
    }
    index_t x = cube.x();
    index_t y = cube.y();
    index_t z = cube.z();
    switch (dim) {
    case 0:
        return {x, y, z};

    case 1:
        switch (cube.type()) {
        case 0:
            if (cube.birth == getBirth(x + 1, y, z)) {
                return {x + 1, y, z};
            } else {
                return {x, y, z};
            }

        case 1:
            if (cube.birth == getBirth(x, y + 1, z)) {
                return {x, y + 1, z};
            } else {
                return {x, y, z};
            }

        case 2:
            if (cube.birth == getBirth(x, y, z + 1)) {
                return {x, y, z + 1};
            } else {
                return {x, y, z};
            }
        }

    case 2:
        switch (cube.type()) {
        case 0:
            if (cube.birth == getBirth(x, y + 1, z + 1)) {
                return {x, y + 1, z + 1};
            } else if (cube.birth == getBirth(x, y + 1, z)) {
                return {x, y + 1, z};
            } else if (cube.birth == getBirth(x, y, z + 1)) {
                return {x, y, z + 1};
            } else {
                return {x, y, z};
            }

        case 1:
            if (cube.birth == getBirth(x + 1, y, z + 1)) {
                return {x + 1, y, z + 1};
            } else if (cube.birth == getBirth(x + 1, y, z)) {
                return {x + 1, y, z};
            } else if (cube.birth == getBirth(x, y, z + 1)) {
                return {x, y, z + 1};
            } else {
                return {x, y, z};
            }

        case 2:
            if (cube.birth == getBirth(x + 1, y + 1, z)) {
                return {x + 1, y + 1, z};
            } else if (cube.birth == getBirth(x + 1, y, z)) {
                return {x + 1, y, z};
            } else if (cube.birth == getBirth(x, y + 1, z)) {
                return {x, y + 1, z};
            } else {
                return {x, y, z};
            }
        }

    case 3:
        if (cube.birth == getBirth(x, y, z)) {
            return {x, y, z};
        } else if (cube.birth == getBirth(x, y, z + 1)) {
            return {x, y, z + 1};
        } else if (cube.birth == getBirth(x, y + 1, z)) {
            return {x, y + 1, z};
        } else if (cube.birth == getBirth(x, y + 1, z + 1)) {
            return {x, y + 1, z + 1};
        } else if (cube.birth == getBirth(x + 1, y, z)) {
            return {x + 1, y, z};
        } else if (cube.birth == getBirth(x + 1, y, z + 1)) {
            return {x + 1, y, z + 1};
        } else if (cube.birth == getBirth(x + 1, y + 1, z)) {
            return {x + 1, y + 1, z};
        } else {
            return {x + 1, y + 1, z + 1};
        }
    }
    throw runtime_error("Parent voxel not found!");
}

void CubicalGridComplex::printImage() const {
    value_t birth;
    for (index_t y = 0; y < shape[1]; ++y) {
        for (index_t x = 0; x < shape[0]; ++x) {
            for (index_t z = 0; z < shape[2]; ++z) {
                birth = getBirth(x, y, z);
                if (birth < 10) {
                    cout << ' ' << birth << ' ';
                } else {
                    cout << birth << ' ';
                }
            }
            cout << "  ";
        }
        cout << endl;
    }
}

void CubicalGridComplex::printRepresentativeCycle(
    const RepresentativeCycle &reprCycle) const {
    for (index_t y = 0; y < shape[1]; ++y) {
        for (index_t x = 0; x < shape[0]; ++x) {
            for (index_t z = 0; z < shape[2]; ++z) {
                auto it = find(reprCycle.begin(), reprCycle.end(),
                               dim3::Coordinate{x, y, z});
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
            cout << "  ";
        }
        cout << endl;
    }
}

void CubicalGridComplex::getGridFromVector(const vector<value_t> &vec) {
#ifdef RUNTIME
    auto _t = RT_NOW();
#endif
    const size_t px = static_cast<size_t>(shape[0]) + 2;
    const size_t py = static_cast<size_t>(shape[1]) + 2;
    const size_t pz = static_cast<size_t>(shape[2]) + 2;
    // Single contiguous allocation; padding (border) stays INFTY, interior
    // is filled row-major in the same order as the input vector.
    grid.assign(px * py * pz, INFTY);
    size_t counter = 0;
    for (index_t x = 1; x <= shape[0]; ++x) {
        for (index_t y = 1; y <= shape[1]; ++y) {
            value_t *row = grid.data() + x * gridStrideX + y * gridStrideY;
            for (index_t z = 1; z <= shape[2]; ++z) {
                row[z] = vec[counter++];
            }
        }
    }
#ifdef RUNTIME
    g_tGrid += RT_SINCE(_t);
#endif
}

// One canonical-order scan of the comparison complex; the kept lists
// replicate the visit order of the full-grid enumerations exactly
// (vertices/edges/2-cells: (x,y,z[,type]) over shape; 3-cells: (x,y,z) over
// (m_x,m_y,m_z) as in the UnionFindDual constructor).
KeptCells::KeptCells(const CubicalGridComplex &cgcComp) {
#ifdef RUNTIME
    auto _t = RT_NOW();
#endif
    const vector<index_t> &shape = cgcComp.shape;
    strideY = static_cast<size_t>(shape[2]) * 3;
    strideX = static_cast<size_t>(shape[1]) * strideY;

    const size_t sx = shape[0], sy = shape[1], sz = shape[2];
    const size_t syz = sy * sz;
    const size_t numVertices = sx * syz;

    // Kept-ness is a BOOLEAN, and it is separable. getBirth(cell) is a max over
    // the cell's corner voxels and the out-of-grid padding is INFTY, so a cell
    // is kept iff every corner voxel is -- and the four classifications are
    // three 2-tap ANDs plus their compositions rather than ~27 float loads per
    // voxel through getBirth.
    //
    //   edge   type d  spans axis d          -> A[d]
    //   2-cell type d  has NORMAL axis d     -> A[the other two]
    //   3-cell                               -> A[xyz]
    vector<uint8_t> K(numVertices, 0);
    {
        size_t i = 0;
        for (index_t x = 0; x < shape[0]; ++x) {
            for (index_t y = 0; y < shape[1]; ++y) {
                for (index_t z = 0; z < shape[2]; ++z, ++i) {
                    K[i] = cgcComp.getBirth(x, y, z) != INFTY ? 1 : 0;
                }
            }
        }
    }
    // B[i] = A[i] & A[i + one step along `axis`], zero on the last slice of
    // that axis (the shifted corner is out of grid, i.e. INFTY, i.e. absent).
    auto shiftAnd = [&](const vector<uint8_t> &A, vector<uint8_t> &B,
                        int axis) {
        B.assign(numVertices, 0);
        const uint8_t *a = A.data();
        uint8_t *b = B.data();
        if (axis == 0) {
            if (sx < 2) { return; }
            const size_t e = (sx - 1) * syz;
            for (size_t i = 0; i < e; ++i) { b[i] = a[i] & a[i + syz]; }
        } else if (axis == 1) {
            if (sy < 2) { return; }
            const size_t e = (sy - 1) * sz;
            for (size_t x = 0; x < sx; ++x) {
                const size_t o = x * syz;
                for (size_t i = 0; i < e; ++i) {
                    b[o + i] = a[o + i] & a[o + i + sz];
                }
            }
        } else {
            if (sz < 2) { return; }
            for (size_t xy = 0; xy < sx * sy; ++xy) {
                const size_t o = xy * sz;
                for (size_t z = 0; z + 1 < sz; ++z) {
                    b[o + z] = a[o + z] & a[o + z + 1];
                }
            }
        }
    };
    vector<uint8_t> Ax, Ay, Az, Axy, Axz, Ayz, Axyz;
    shiftAnd(K, Ax, 0);
    shiftAnd(K, Ay, 1);
    shiftAnd(K, Az, 2);
    shiftAnd(Ax, Axy, 1);
    shiftAnd(Ax, Axz, 2);
    shiftAnd(Ay, Ayz, 2);
    shiftAnd(Axy, Axyz, 2);
    auto popcount = [](const vector<uint8_t> &A) {
        size_t c = 0;
        for (const uint8_t v : A) { c += v; }
        return c;
    };

    // Vertices: slot = dense vertex id (x*n_yz + y*n_z + z), the id space of
    // the dim-0 UnionFind.
    compactVertex.assign(numVertices, -1);
    keptVertices.reserve(popcount(K));
    for (size_t slot = 0; slot < numVertices; ++slot) {
        if (K[slot]) {
            compactVertex[slot] = static_cast<int32_t>(keptVertices.size());
            keptVertices.push_back(static_cast<index_t>(slot));
        }
    }

    // Edges and 2-cells: the (x,y,z,type) CubeMap slot space (3N slots), in the
    // same canonical (x,y,z,type) order as the full-grid enumerations.
    compactEdge.assign(numVertices * 3, -1);
    compactTwoCell.assign(numVertices * 3, -1);
    keptEdges.reserve(popcount(Ax) + popcount(Ay) + popcount(Az));
    keptTwoCells.reserve(popcount(Ayz) + popcount(Axz) + popcount(Axy));
    {
        const uint8_t *E[3] = {Ax.data(), Ay.data(), Az.data()};
        const uint8_t *F[3] = {Ayz.data(), Axz.data(), Axy.data()};
        size_t slot = 0, v = 0;
        for (index_t x = 0; x < shape[0]; ++x) {
            for (index_t y = 0; y < shape[1]; ++y) {
                for (index_t z = 0; z < shape[2]; ++z, ++v) {
                    const uint64_t base = ((uint64_t)x << 44) |
                                          ((uint64_t)y << 24) |
                                          ((uint64_t)z << 4);
                    for (uint8_t type = 0; type < 3; ++type, ++slot) {
                        if (E[type][v]) {
                            compactEdge[slot] =
                                static_cast<int32_t>(keptEdges.size());
                            keptEdges.push_back(base | (uint64_t)type);
                        }
                        if (F[type][v]) {
                            compactTwoCell[slot] =
                                static_cast<int32_t>(keptTwoCells.size());
                            keptTwoCells.push_back(base | (uint64_t)type);
                        }
                    }
                }
            }
        }
    }

    // 3-cells: the UnionFindDual node id space (x*m_yz + y*m_z + z); the
    // exterior sentinel (m_xyz) is handled by the compact dual UF itself.
    const size_t numTopCells = cgcComp.getNumberOfCubes(3);
    compactTopCell.assign(numTopCells, -1);
    keptTopCells.reserve(popcount(Axyz));
    {
        const uint8_t *T = Axyz.data();
        size_t node = 0;
        for (index_t x = 0; x < cgcComp.m_x; ++x) {
            for (index_t y = 0; y < cgcComp.m_y; ++y) {
                const size_t o = static_cast<size_t>(x) * syz +
                                 static_cast<size_t>(y) * sz;
                for (index_t z = 0; z < cgcComp.m_z; ++z, ++node) {
                    if (T[o + z]) {
                        compactTopCell[node] =
                            static_cast<int32_t>(keptTopCells.size());
                        keptTopCells.push_back(static_cast<index_t>(node));
                    }
                }
            }
        }
    }
#ifdef RUNTIME
    g_tKept += RT_SINCE(_t);
#endif
}

PocketNodes::PocketNodes(const CubicalGridComplex &cgcComp,
                         const vector<int32_t> &voxelLabels, size_t keptVoxels)
    : labels(&voxelLabels) {
    const index_t sx = cgcComp.shape[0];
    const index_t sy = cgcComp.shape[1];
    sz = static_cast<size_t>(cgcComp.shape[2]);
    syz = static_cast<size_t>(sy) * sz;
    mx = cgcComp.m_x;
    my = cgcComp.m_y;
    mz = cgcComp.m_z;
    myz = static_cast<size_t>(cgcComp.m_yz);
    const size_t expected = static_cast<size_t>(sx) * syz;
    if (voxelLabels.size() != expected) {
        throw runtime_error("pocket_labels size does not match the input shape");
    }
#ifdef RUNTIME
    auto _tv = RT_NOW();
#endif
    // ONE fused O(N) pass. The label range and the mask agreement are both
    // functions of (voxel, label), so they cost a single traversal.
    //
    // The mask check is not optional: without it a mismatched labelling (wrong
    // sample in a batch, a stale array, a mask recomputed at a different tau)
    // is accepted silently -- a masked cube whose label reads 0 resolves to the
    // exterior sentinel and the cavity simply disappears.
    {
        const int32_t *L = voxelLabels.data();
        size_t i = 0;
        for (index_t x = 0; x < sx; ++x) {
            for (index_t y = 0; y < sy; ++y) {
                for (index_t z = 0; z < static_cast<index_t>(sz); ++z, ++i) {
                    const int32_t v = L[i];
                    if (v < 0) {
                        throw runtime_error(
                            "pocket_labels must be non-negative");
                    }
                    if (v > n) {
                        n = v;
                    }
                    if ((v != 0) != (cgcComp.getBirth(x, y, z) == INFTY)) {
                        throw runtime_error(
                            "pocket_labels support does not match the mask: "
                            "labels must be nonzero exactly on the masked "
                            "(INFTY) voxels");
                    }
                }
            }
        }
    }
#ifdef RUNTIME
    g_tPocketVal += RT_SINCE(_tv);
    auto _tt = RT_NOW();
#endif
    // Boundary-touch flags: a pocket IS the exterior iff it has a masked voxel
    // on a grid face, so only the six faces need visiting (O(N^2/3) voxels).
    boundary.assign(static_cast<size_t>(n) + 1, 0);
    {
        const int32_t *L = voxelLabels.data();
        auto mark = [&](size_t i) {
            const int32_t lab = L[i];
            if (lab > 0) {
                boundary[lab] = 1;
            }
        };
        for (index_t y = 0; y < sy; ++y) {
            for (index_t z = 0; z < static_cast<index_t>(sz); ++z) {
                const size_t off = static_cast<size_t>(y) * sz + z;
                mark(off);
                mark(static_cast<size_t>(sx - 1) * syz + off);
            }
        }
        for (index_t x = 0; x < sx; ++x) {
            const size_t ox = static_cast<size_t>(x) * syz;
            for (index_t z = 0; z < static_cast<index_t>(sz); ++z) {
                mark(ox + z);
                mark(ox + static_cast<size_t>(sy - 1) * sz + z);
            }
            for (index_t y = 0; y < sy; ++y) {
                const size_t oy = ox + static_cast<size_t>(y) * sz;
                mark(oy);
                mark(oy + sz - 1);
            }
        }
    }
    // See the header: the cube labelling is memoised only where the probe
    // count justifies the 8N-load build.
    if (26 * keptVoxels > static_cast<size_t>(sx) * syz) {
        // Fill a local and swap: ofTopCellAt() consults cubePocket first, so
        // populating it in place would have it read its own zeros.
        vector<int32_t> memo(static_cast<size_t>(cgcComp.m_xyz), 0);
        for (index_t x = 0; x < mx; ++x) {
            for (index_t y = 0; y < my; ++y) {
                for (index_t z = 0; z < mz; ++z) {
                    memo[static_cast<size_t>(x) * myz +
                         static_cast<size_t>(y) * mz + z] = ofTopCellAt(x, y, z);
                }
            }
        }
        cubePocket.swap(memo);
    }
#ifdef RUNTIME
    g_tPocketTop += RT_SINCE(_tt);
#endif
}

ConeIndex::ConeIndex(const CubicalGridComplex &cgcComp, const KeptCells &kept,
                     const PocketNodes &pockets, bool prune) {
#ifdef RUNTIME
    auto _te = RT_NOW();
#endif
    const vector<index_t> &shape = cgcComp.shape;
    const index_t nx = shape[0], ny = shape[1], nz = shape[2];
    const index_t mx = cgcComp.m_x, my = cgcComp.m_y, mz = cgcComp.m_z;

    // Apex of the non-kept top cell with min-corner (cx,cy,cz), 0 if out of
    // grid or kept.
    //
    // This is the FINE labelling (26-connected masked voxels), and that is the
    // derived answer, not a preference. What the dim-1 reduction needs is that
    // splitting the filling 2-chain across apexes keeps each piece's boundary
    // inside the kept complex -- i.e. every non-kept 1-cell must have all of
    // its cofacet 2-cells under ONE apex. A non-kept 1-cell has a masked
    // vertex m, and every 2-cell containing that edge also contains m, so
    // grouping by "shares a non-kept face" suffices -- and those components are
    // exactly the pocket labelling itself, the 26-connected components of the
    // masked VOXELS.
    //
    // That is also the ONLY hypothesis the correctness argument needs: no
    // non-kept cell shared between two apexes. It lets the pockets be filled
    // one at a time: with X_j = X_{j-1} u N_j, the intersection
    // X_{j-1} n N_j is exactly the rim of pocket j, so by Mayer-Vietoris (the
    // box being acyclic) the cones together kill all of H_n(K). No condition
    // on a pocket's own topology, e.g. contractibility, is needed.
    auto cubeApex = [&](index_t cx, index_t cy, index_t cz) -> int32_t {
        if (cx >= mx || cy >= my || cz >= mz) {
            return 0; // unsigned wrap covers the negative case
        }
        return pockets.ofTopCellAt(cx, cy, cz);
    };
    // Distinct, sorted pockets over a set of cube probes. At most 8, so
    // an insertion sort beats any container.
    auto collect = [](int32_t *buf, int32_t &count, int32_t p) {
        if (p == 0) {
            return;
        }
        for (int32_t i = 0; i < count; ++i) {
            if (buf[i] == p) {
                return;
            }
            if (buf[i] > p) {
                for (int32_t j = count; j > i; --j) {
                    buf[j] = buf[j - 1];
                }
                buf[i] = p;
                ++count;
                return;
            }
        }
        buf[count++] = p;
    };

    // --- rim vertices: the <= 8 cubes with min-corner in {x-1,x} x ... ------
    vStart.assign(kept.keptVertices.size() + 1, 0);
    vPocket.reserve(kept.keptVertices.size());
    for (size_t i = 0; i < kept.keptVertices.size(); ++i) {
        const index_t slot = kept.keptVertices[i];
        const index_t x = slot / (ny * nz);
        const index_t y = (slot / nz) % ny;
        const index_t z = slot % nz;
        int32_t buf[8];
        int32_t count = 0;
        for (int dx = -1; dx <= 0; ++dx) {
            for (int dy = -1; dy <= 0; ++dy) {
                for (int dz = -1; dz <= 0; ++dz) {
                    if ((dx < 0 && x == 0) || (dy < 0 && y == 0) ||
                        (dz < 0 && z == 0)) {
                        continue;
                    }
                    collect(buf, count, cubeApex(x + dx, y + dy, z + dz));
                }
            }
        }
        vStart[i + 1] = vStart[i] + count;
        for (int32_t j = 0; j < count; ++j) {
            vPocket.push_back(buf[j]);
        }
    }

    // --- rim edges: the <= 4 cubes containing the edge ----------------------
    // Edge (x,y,z,type) spans axis `type`, so a containing cube has
    // c[type] == x[type] and c[d] in {x[d]-1, x[d]} for d != type.
    auto rankAt = [&](index_t x, index_t y, index_t z, int32_t pocket) -> uint8_t {
        const int32_t v = kept.vertexCompact(x * ny * nz + y * nz + z);
        for (int32_t r = vStart[v]; r < vStart[v + 1]; ++r) {
            if (vPocket[r] == pocket) {
                return static_cast<uint8_t>(r - vStart[v]);
            }
        }
        throw runtime_error(
            "ConeIndex: a rim edge's endpoint is not a rim vertex of the same "
            "pocket -- the cone would not close");
    };

    eStart.assign(kept.keptEdges.size() + 1, 0);
    for (size_t i = 0; i < kept.keptEdges.size(); ++i) {
        const uint64_t key = kept.keptEdges[i];
        const index_t x = (key >> 44) & 0xfffff;
        const index_t y = (key >> 24) & 0xfffff;
        const index_t z = (key >> 4) & 0xfffff;
        const uint8_t type = key & 0xf;
        index_t base[3] = {x, y, z};
        int32_t buf[4];
        int32_t count = 0;
        for (int a = -1; a <= 0; ++a) {
            for (int b = -1; b <= 0; ++b) {
                index_t c[3] = {base[0], base[1], base[2]};
                int which = 0;
                bool ok = true;
                for (int d = 0; d < 3; ++d) {
                    if (d == type) {
                        continue;
                    }
                    const int off = (which++ == 0) ? a : b;
                    if (off < 0 && c[d] == 0) {
                        ok = false;
                        break;
                    }
                    c[d] += off;
                }
                if (ok) {
                    collect(buf, count, cubeApex(c[0], c[1], c[2]));
                }
            }
        }
        eStart[i + 1] = eStart[i] + count;
        index_t v1[3] = {x, y, z};
        ++v1[type];
        for (int32_t j = 0; j < count; ++j) {
            ePocket.push_back(buf[j]);
            eRankV0.push_back(rankAt(x, y, z, buf[j]));
            eRankV1.push_back(rankAt(v1[0], v1[1], v1[2], buf[j]));
        }
    }

    // Pruning needs the rim FACES and the rim-edge -> rim-face incidence; the
    // unpruned path does not, so only pay for them when asked.
#ifdef RUNTIME
    g_tConeVE += RT_SINCE(_te);
#endif
    if (prune) {
#ifdef RUNTIME
        auto _tf = RT_NOW();
#endif
        buildFaces(cgcComp, kept, pockets);
#ifdef RUNTIME
        g_tConeFaces += RT_SINCE(_tf);
        auto _ti = RT_NOW();
#endif
        buildEdgeFaceIncidence(cgcComp, kept);
#ifdef RUNTIME
        g_tConeInc += RT_SINCE(_ti);
#endif
    }
#ifdef RUNTIME
    auto _tm = RT_NOW();
#endif
    computeEmitMask(cgcComp, kept, prune);
#ifdef RUNTIME
    g_tConeEmit += RT_SINCE(_tm);
#endif
    // Build with -DNO_CONE_CYCLES to emit the cone without explicit cycles
    // (one column per emitted rim edge, plus cone-edge rows); for timing
    // comparisons only.
#ifndef NO_CONE_CYCLES
    if (prune) {
#ifdef RUNTIME
        auto _tc = RT_NOW();
#endif
        buildCycles(cgcComp, kept);
#ifdef RUNTIME
        g_tConeCycles += RT_SINCE(_tc);
#endif
    }
#endif
}

// The spanning forest of the rim graph, and one explicit fundamental cycle per
// non-tree emitted column. See the comment on ConeIndex::hasCycles().
void ConeIndex::buildCycles(const CubicalGridComplex &cgcComp,
                            const KeptCells &kept) {
    const size_t E = ePocket.size();
    const size_t V = vPocket.size();
    cycIndex.assign(E, -1);
    cycPtr.assign(1, 0);
    if (E == 0 || V == 0) {
        return;
    }

    // Endpoints (cone-edge slots) and the anchor kept edge of every rim-edge
    // slot. Recomputed rather than stored, exactly as computeEmitMask does.
    const index_t ny = cgcComp.shape[1], nz = cgcComp.shape[2];
    vector<int32_t> endV0(E), endV1(E);
    vector<uint64_t> anchor(E);
    for (size_t i = 0; i + 1 < eStart.size(); ++i) {
        const uint64_t key = kept.keptEdges[i];
        const index_t x = (key >> 44) & 0xfffff;
        const index_t y = (key >> 24) & 0xfffff;
        const index_t z = (key >> 4) & 0xfffff;
        const uint8_t type = key & 0xf;
        index_t v1[3] = {x, y, z};
        ++v1[type];
        const int32_t c0 = kept.vertexCompact(x * ny * nz + y * nz + z);
        const int32_t c1 =
            kept.vertexCompact(v1[0] * ny * nz + v1[1] * nz + v1[2]);
        for (int32_t slot = eStart[i]; slot < eStart[i + 1]; ++slot) {
            endV0[slot] = coneEdgeSlot(c0, eRankV0[slot]);
            endV1[slot] = coneEdgeSlot(c1, eRankV1[slot]);
            anchor[slot] = key;
        }
    }

    // --- the forest, greedily in SLOT order (== the reduction's order) ------
    vector<int32_t> uf(V);
    for (size_t i = 0; i < V; ++i) {
        uf[i] = static_cast<int32_t>(i);
    }
    auto find = [&uf](int32_t x) {
        while (uf[x] != x) {
            uf[x] = uf[uf[x]]; // path halving
            x = uf[x];
        }
        return x;
    };
    vector<int32_t> treeSlots;
    vector<int32_t> cycleSlots;
    treeSlots.reserve(V);
    for (size_t e = 0; e < E; ++e) {
        if (!emitCone2Cell(static_cast<int32_t>(e))) {
            continue;
        }
        const int32_t a = find(endV0[e]), b = find(endV1[e]);
        if (a != b) {
            uf[a] = b;
            treeSlots.push_back(static_cast<int32_t>(e));
        } else {
            cycleSlots.push_back(static_cast<int32_t>(e));
        }
    }
    if (cycleSlots.empty()) {
        return;
    }

    // --- root the forest: parent node, parent slot, depth -------------------
    // A path in a forest is unique and never changes as further tree edges are
    // added, so rooting the FINAL forest gives exactly the path that existed
    // when the non-tree column was reached.
    const size_t T = treeSlots.size();
    vector<int32_t> ptr(V + 1, 0);
    for (const int32_t s : treeSlots) {
        ++ptr[endV0[s] + 1];
        ++ptr[endV1[s] + 1];
    }
    for (size_t i = 0; i < V; ++i) {
        ptr[i + 1] += ptr[i];
    }
    vector<int32_t> fill(ptr.begin(), ptr.end() - 1);
    vector<int32_t> adjNode(2 * T), adjSlot(2 * T);
    for (const int32_t s : treeSlots) {
        const int32_t a = endV0[s], b = endV1[s];
        adjNode[fill[a]] = b;
        adjSlot[fill[a]] = s;
        ++fill[a];
        adjNode[fill[b]] = a;
        adjSlot[fill[b]] = s;
        ++fill[b];
    }
    vector<int32_t> par(V, -1), parSlot(V, -1), depth(V, -1);
    vector<int32_t> stack;
    stack.reserve(V);
    for (size_t r = 0; r < V; ++r) {
        if (depth[r] >= 0 || ptr[r + 1] == ptr[r]) {
            continue; // already visited, or an isolated node (no tree edge)
        }
        depth[r] = 0;
        stack.push_back(static_cast<int32_t>(r));
        while (!stack.empty()) {
            const int32_t x = stack.back();
            stack.pop_back();
            for (int32_t k = ptr[x]; k < ptr[x + 1]; ++k) {
                const int32_t y = adjNode[k];
                if (depth[y] < 0) {
                    depth[y] = depth[x] + 1;
                    par[y] = x;
                    parSlot[y] = adjSlot[k];
                    stack.push_back(y);
                }
            }
        }
    }

    // --- one explicit fundamental cycle per non-tree column -----------------
    cycPtr.assign(cycleSlots.size() + 1, 0);
    vector<uint64_t> buf;
    for (size_t c = 0; c < cycleSlots.size(); ++c) {
        const int32_t s = cycleSlots[c];
        cycIndex[static_cast<size_t>(s)] = static_cast<int32_t>(c);
        buf.clear();
        buf.push_back(anchor[s]);
        int32_t a = endV0[s], b = endV1[s];
        while (depth[a] > depth[b]) {
            buf.push_back(anchor[parSlot[a]]);
            a = par[a];
        }
        while (depth[b] > depth[a]) {
            buf.push_back(anchor[parSlot[b]]);
            b = par[b];
        }
        while (a != b) {
            buf.push_back(anchor[parSlot[a]]);
            buf.push_back(anchor[parSlot[b]]);
            a = par[a];
            b = par[b];
        }
        // A fundamental cycle of a simple graph has no repeated edge: the rim
        // graph splits by pocket, and within one pocket a rim edge appears
        // once. Sorting DESCENDING is what hasPreviousFace() consumes.
        sort(buf.begin(), buf.end(), std::greater<uint64_t>());
        cycPtr[c + 1] = cycPtr[c] + static_cast<int32_t>(buf.size());
        cycEdge.insert(cycEdge.end(), buf.begin(), buf.end());
    }
}

// A kept 2-cell (x,y,z,type) has `type` as its NORMAL axis -- type 0 spans
// (y,z), 1 spans (x,z), 2 spans (x,y); see getParentVoxel's `case 2`, whose
// type-0 branch probes (x, y+1, z+1). Its two cofacet cubes therefore differ
// only along `type`. Probing a SPANNED axis instead is the trap here: it
// yields a plausible but wrong face count, and the rim-closure check below
// does not catch it on single-pocket volumes.
void ConeIndex::buildFaces(const CubicalGridComplex &cgcComp,
                           const KeptCells &kept, const PocketNodes &pockets) {
    const index_t mx = cgcComp.m_x, my = cgcComp.m_y, mz = cgcComp.m_z;
    auto cubePocket = [&](index_t cx, index_t cy, index_t cz) -> int32_t {
        if (cx >= mx || cy >= my || cz >= mz) {
            return 0; // unsigned wrap covers the negative case
        }
        return pockets.ofTopCellAt(cx, cy, cz);
    };
    fStart.assign(kept.keptTwoCells.size() + 1, 0);
    for (size_t i = 0; i < kept.keptTwoCells.size(); ++i) {
        const uint64_t key = kept.keptTwoCells[i];
        index_t c[3] = {static_cast<index_t>((key >> 44) & 0xfffff),
                        static_cast<index_t>((key >> 24) & 0xfffff),
                        static_cast<index_t>((key >> 4) & 0xfffff)};
        const uint8_t type = key & 0xf;
        int32_t buf[2];
        int32_t count = 0;
        for (int off = -1; off <= 0; ++off) {
            if (off < 0 && c[type] == 0) {
                continue;
            }
            index_t p[3] = {c[0], c[1], c[2]};
            p[type] += off;
            const int32_t q = cubePocket(p[0], p[1], p[2]);
            if (q != 0 && (count == 0 || buf[0] != q)) {
                buf[count++] = q;
            }
        }
        if (count == 2 && buf[0] > buf[1]) {
            std::swap(buf[0], buf[1]);
        }
        fStart[i + 1] = fStart[i] + count;
        for (int32_t j = 0; j < count; ++j) {
            fPocket.push_back(buf[j]);
        }
    }
}

// Each rim edge lies in <= 4 grid 2-cells: for each axis d != the edge's own
// axis, the 2-cell whose NORMAL is the remaining axis, at the edge's corner
// and at the corner shifted -1 along d. Only faces that are rim faces of the
// SAME pocket count -- that is what makes the boundary row well defined.
void ConeIndex::buildEdgeFaceIncidence(const CubicalGridComplex &cgcComp,
                                       const KeptCells &kept) {
    efStart.assign(ePocket.size() + 1, 0);
    // Two passes: count, then fill, so the CSR needs no per-slot vector.
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            for (size_t i = 0; i < ePocket.size(); ++i) {
                efStart[i + 1] += efStart[i];
            }
            efFace.assign(efStart.back(), 0);
        }
        vector<int32_t> fill;
        if (pass == 1) {
            fill.assign(efStart.begin(), efStart.end() - 1);
        }
        for (size_t i = 0; i < kept.keptEdges.size(); ++i) {
            const uint64_t key = kept.keptEdges[i];
            index_t base[3] = {static_cast<index_t>((key >> 44) & 0xfffff),
                               static_cast<index_t>((key >> 24) & 0xfffff),
                               static_cast<index_t>((key >> 4) & 0xfffff)};
            const uint8_t etype = key & 0xf;
            for (int32_t slot = eStart[i]; slot < eStart[i + 1]; ++slot) {
                const int32_t pocket = ePocket[slot];
                for (uint8_t d = 0; d < 3; ++d) {
                    if (d == etype) {
                        continue;
                    }
                    const uint8_t normal = 3 - etype - d;
                    for (int off = -1; off <= 0; ++off) {
                        if (off < 0 && base[d] == 0) {
                            continue;
                        }
                        index_t f[3] = {base[0], base[1], base[2]};
                        f[d] += off;
                        const uint64_t fkey =
                            (static_cast<uint64_t>(f[0]) << 44) |
                            (static_cast<uint64_t>(f[1]) << 24) |
                            (static_cast<uint64_t>(f[2]) << 4) | normal;
                        const int32_t fc = kept.twoCellCompact(fkey);
                        if (fc < 0) {
                            continue; // not a kept 2-cell
                        }
                        for (int32_t fs = fStart[fc]; fs < fStart[fc + 1];
                             ++fs) {
                            if (fPocket[fs] != pocket) {
                                continue;
                            }
                            if (pass == 0) {
                                ++efStart[slot + 1];
                            } else {
                                efFace[fill[slot]++] = fs;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Greedy over the matroid of boundary ROWS (row_e = indicator of the rim faces
// containing e). Any independent set R may be DROPPED: each dropped cycle then
// bounds across rim 2-cells, which are real columns of the matrix already, so
// the cone kills exactly the same subspace of H_1(K). Greedy in any order
// attains the optimum |R| = rank d_2(Rim), leaving b_1(Rim) expensive columns.
//
// Two stages, because the degree-2 rows are a graph and deserve a union-find:
//   1. rows of weight 2 are dual-graph edges -> independence == acyclicity;
//   2. every other row is projected into the quotient GF(2)^{C*} (C* = dual
//      components left after stage 1) and eliminated there.
// Stage 2 is what reaches the optimum.
void ConeIndex::computeEmitMask(const CubicalGridComplex &cgcComp,
                                const KeptCells &kept, bool prune) {
    const size_t E = ePocket.size();
    emitBits.assign((E + 63) / 64, ~0ULL);
    nEmitted = E;
    if (!prune || E == 0) {
        return;
    }

    // Cone-edge slot of each endpoint of each rim-edge slot. Recomputed here
    // rather than stored, so the unpruned path pays nothing for it.
    const index_t ny = cgcComp.shape[1], nz = cgcComp.shape[2];
    vector<int32_t> endV0(E), endV1(E);
    for (size_t i = 0; i + 1 < eStart.size(); ++i) {
        const uint64_t key = kept.keptEdges[i];
        const index_t x = (key >> 44) & 0xfffff;
        const index_t y = (key >> 24) & 0xfffff;
        const index_t z = (key >> 4) & 0xfffff;
        const uint8_t type = key & 0xf;
        index_t v1[3] = {x, y, z};
        ++v1[type];
        const int32_t c0 = kept.vertexCompact(x * ny * nz + y * nz + z);
        const int32_t c1 =
            kept.vertexCompact(v1[0] * ny * nz + v1[1] * nz + v1[2]);
        for (int32_t slot = eStart[i]; slot < eStart[i + 1]; ++slot) {
            endV0[slot] = coneEdgeSlot(c0, eRankV0[slot]);
            endV1[slot] = coneEdgeSlot(c1, eRankV1[slot]);
        }
    }

    const size_t F = fPocket.size();
    vector<int32_t> parent(F);
    for (size_t i = 0; i < F; ++i) {
        parent[i] = static_cast<int32_t>(i);
    }
    auto find = [&parent](int32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        return x;
    };
    vector<bool> redundant(E, false);

    // --- stage 1: the weight-2 rows, as a spanning forest of the dual graph
    for (size_t e = 0; e < E; ++e) {
        if (efStart[e + 1] - efStart[e] != 2) {
            continue;
        }
        const int32_t a = find(efFace[efStart[e]]);
        const int32_t b = find(efFace[efStart[e] + 1]);
        if (a != b) {
            parent[a] = b;
            redundant[e] = true;
        }
    }

    // --- stage 2: every other row, in the quotient by stage 1 --------------
    // Quotienting by the span of a spanning forest's edge vectors leaves
    // exactly "parity within each dual component", i.e. GF(2)^{C*}.
    vector<int32_t> compOf(F, -1);
    int32_t nComp = 0;
    for (size_t i = 0; i < F; ++i) {
        const int32_t r = find(static_cast<int32_t>(i));
        if (compOf[r] < 0) {
            compOf[r] = nComp++;
        }
    }
    if (nComp > 0) {
        const size_t words = (static_cast<size_t>(nComp) + 63) / 64;
        vector<uint64_t> row(words);
        vector<size_t> pivotAt(nComp, SIZE_MAX); // component -> offset in basis
        vector<uint64_t> basis;
        for (size_t e = 0; e < E; ++e) {
            const int32_t lo = efStart[e], hi = efStart[e + 1];
            if (hi - lo == 2) {
                continue; // stage 1 owns these, and they project to zero here
            }
            std::fill(row.begin(), row.end(), 0ULL);
            for (int32_t k = lo; k < hi; ++k) {
                // XOR, not OR: two incident faces in one component cancel,
                // which is precisely the GF(2) projection.
                const size_t c = static_cast<size_t>(compOf[find(efFace[k])]);
                row[c >> 6] ^= 1ULL << (c & 63);
            }
            for (;;) {
                size_t w = 0;
                while (w < words && row[w] == 0ULL) {
                    ++w;
                }
                if (w == words) {
                    break; // dependent -- this column has to stay
                }
                const size_t bit =
                    (w << 6) + static_cast<size_t>(__builtin_ctzll(row[w]));
                if (pivotAt[bit] == SIZE_MAX) {
                    pivotAt[bit] = basis.size();
                    basis.insert(basis.end(), row.begin(), row.end());
                    redundant[e] = true;
                    break;
                }
                const uint64_t *b = basis.data() + pivotAt[bit];
                for (size_t j = w; j < words; ++j) {
                    row[j] ^= b[j];
                }
            }
        }
    }

    nEmitted = 0;
    for (size_t e = 0; e < E; ++e) {
        if (redundant[e]) {
            emitBits[e >> 6] &= ~(1ULL << (e & 63));
        } else {
            ++nEmitted;
        }
    }

    // FREE-SPANNING invariant, checked rather than trusted: an independent row
    // set contains no edge cut of the rim graph (every boundary is a cycle and
    // meets a cut evenly), so the surviving edges must still span it. If they
    // did not, the cone-edge row block would lose rank and spurious unpaired
    // cone edges would appear at CONE_BIRTH. O(E alpha), a few ms against a cone
    // block of seconds.
    {
        const size_t V = vPocket.size();
        vector<int32_t> vp(V);
        auto vfind = [&vp](int32_t x) {
            while (vp[x] != x) {
                vp[x] = vp[vp[x]];
                x = vp[x];
            }
            return x;
        };
        auto components = [&](bool emittedOnly) {
            for (size_t i = 0; i < V; ++i) {
                vp[i] = static_cast<int32_t>(i);
            }
            size_t merges = 0;
            for (size_t e = 0; e < E; ++e) {
                if (emittedOnly && !emitCone2Cell(static_cast<int32_t>(e))) {
                    continue;
                }
                const int32_t a = vfind(endV0[e]), b = vfind(endV1[e]);
                if (a != b) {
                    vp[a] = b;
                    ++merges;
                }
            }
            return merges;
        };
        const size_t full = components(false);
        const size_t kept2 = components(true);
        if (full != kept2) {
            throw runtime_error(
                "cone pruning violated the free-spanning invariant: the "
                "emitted rim edges realise " +
                std::to_string(kept2) + " merges against " +
                std::to_string(full) + " for the full rim graph");
        }
    }
}

UnionFind::UnionFind(const CubicalGridComplex &_cgc, const KeptCells *_kept)
    : cgc(_cgc), kept(_kept) {
    if (kept == nullptr) {
        index_t n = cgc.getNumberOfCubes(0);
        parent.reserve(n);
        birthtime.reserve(n);
        index_t counter = 0;
        for (index_t x = 0; x < _cgc.shape[0]; ++x) {
            for (index_t y = 0; y < _cgc.shape[1]; ++y) {
                for (index_t z = 0; z < _cgc.shape[2]; ++z) {
                    parent.push_back(counter++);
                    birthtime.push_back(cgc.getBirth(x, y, z));
                }
            }
        }
        return;
    }
    // Compact (sparse) mode: nodes are the kept vertices, in canonical order
    // (which is increasing dense vertex id).
    const size_t n = kept->keptVertices.size();
    parent.reserve(n);
    birthtime.reserve(n);
    original = kept->keptVertices;
    index_t counter = 0;
    for (const index_t vertexIdx : kept->keptVertices) {
        parent.push_back(counter++);
        const index_t x = vertexIdx / cgc.n_yz;
        const index_t y = (vertexIdx / cgc.shape[2]) % cgc.shape[1];
        const index_t z = vertexIdx % cgc.shape[2];
        birthtime.push_back(cgc.getBirth(x, y, z));
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
        // Equal-birth tie-break on the ORIGINAL dense vertex index:
        // identical to upstream in dense mode; in compact mode the compact
        // ids must never be compared directly.
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

Coordinate UnionFind::getCoordinates(index_t idx) const {
    if (kept != nullptr) {
        idx = original[idx];
    }
    return {idx / cgc.n_yz, idx / cgc.shape[2] % cgc.shape[1],
            idx % cgc.shape[2]};
}

vector<index_t> UnionFind::getBoundaryIndices(const Cube &edge) const {
    vector<index_t> boundaryIndices(2);
    switch (edge.type()) {
    case 0:
        boundaryIndices[0] =
            edge.x() * cgc.n_yz + edge.y() * cgc.shape[2] + edge.z();
        boundaryIndices[1] =
            (edge.x() + 1) * cgc.n_yz + edge.y() * cgc.shape[2] + edge.z();
        break;

    case 1:
        boundaryIndices[0] =
            edge.x() * cgc.n_yz + edge.y() * cgc.shape[2] + edge.z();
        boundaryIndices[1] =
            edge.x() * cgc.n_yz + (edge.y() + 1) * cgc.shape[2] + edge.z();
        break;

    case 2:
        boundaryIndices[0] =
            edge.x() * cgc.n_yz + edge.y() * cgc.shape[2] + edge.z();
        boundaryIndices[1] =
            edge.x() * cgc.n_yz + edge.y() * cgc.shape[2] + edge.z() + 1;
        break;
    }
    if (kept != nullptr) {
        // A kept edge has kept endpoints (shared mask), so both compact
        // ids exist.
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
        index_t n = cgc.getNumberOfCubes(3) + 1;
        parent.reserve(n);
        birthtime.reserve(n);
        index_t counter = 0;
        for (index_t x = 0; x < _cgc.m_x; ++x) {
            for (index_t y = 0; y < _cgc.m_y; ++y) {
                for (index_t z = 0; z < _cgc.m_z; ++z) {
                    parent.push_back(counter++);
                    birthtime.push_back(cgc.getBirth(x, y, z, 0, 3));
                }
            }
        }
        parent.push_back(counter);
        birthtime.push_back(INFTY);
        sentinelIdx = static_cast<index_t>(parent.size() - 1);
        return;
    }
    // Compact (sparse) mode: nodes are the kept top-cells in canonical order
    // (increasing dense node id), the 8-corner birth computed only for
    // them, plus the exterior sentinel as the LAST node (original id
    // m_xyz -- larger than every real node id, as in the dense layout).
    const size_t n = kept->keptTopCells.size() + 1;
    parent.reserve(n);
    birthtime.reserve(n);
    original.reserve(n);
    index_t counter = 0;
    for (const index_t nodeIdx : kept->keptTopCells) {
        parent.push_back(counter++);
        const index_t x = nodeIdx / cgc.m_yz;
        const index_t y = (nodeIdx / cgc.m_z) % cgc.m_y;
        const index_t z = nodeIdx % cgc.m_z;
        birthtime.push_back(cgc.getBirth(x, y, z, 0, 3));
        original.push_back(nodeIdx);
    }
    parent.push_back(counter);
    birthtime.push_back(INFTY);
    original.push_back(cgc.m_xyz);
    sentinelIdx = static_cast<index_t>(parent.size() - 1);
    if (pockets != nullptr) {
        // One virtual node per pocket at CONE_BIRTH -- the value the masked
        // region carries in every complex. It exceeds every kept top cell's
        // value, so a pocket is elder to all of them (the dual is swept in
        // decreasing order, so link() lets the SMALLER birthtime die) and can
        // only die against the sentinel or another pocket -- exactly the
        // genuine cavity deaths.
        for (int32_t p = 1; p <= pockets->numPockets(); ++p) {
            parent.push_back(static_cast<index_t>(parent.size()));
            birthtime.push_back(pocketBirth);
            original.push_back(cgc.m_xyz + p);
        }
    }
}

void UnionFindDual::seedPockets() {
    if (pockets == nullptr) {
        return;
    }
    // Boundary-touching pockets ARE the exterior: pre-union them with the
    // sentinel (eldest at INFTY, so it never dies) and they emit no bar --
    // matching the dense semantics where a void reaching the exterior is not
    // a cavity.
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
        // Equal-birth tie-break on the ORIGINAL dense node index:
        // identical to upstream in dense mode.
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

Coordinate UnionFindDual::getCoordinates(index_t x) const {
    if (isPocketNode(x) || x == sentinelIdx) {
        // Virtual: no voxel exists. Reaching here means a censored or exterior
        // root leaked into a voxel-reporting path -- always a bug, never a
        // fallback, so fail loudly rather than return a plausible coordinate.
        throw runtime_error(
            "UnionFindDual::getCoordinates on a virtual node (pocket or "
            "exterior sentinel): censored bars must be reported by their "
            "birth cell");
    }
    if (kept != nullptr) {
        x = original[x];
    }
    return {x / cgc.m_yz, x / cgc.m_z % cgc.m_y, x % cgc.m_z};
}

vector<index_t> UnionFindDual::getBoundaryIndices(const Cube &edge) const {
    vector<index_t> boundaryIndices(2);
    switch (edge.type()) {
    case 0:
        if (edge.x() == 0) {
            boundaryIndices[0] = cgc.m_xyz;
        } else {
            boundaryIndices[0] =
                (edge.x() - 1) * cgc.m_yz + edge.y() * cgc.m_z + edge.z();
        }
        if (edge.x() == cgc.m_x) {
            boundaryIndices[1] = cgc.m_xyz;
        } else {
            boundaryIndices[1] =
                edge.x() * cgc.m_yz + edge.y() * cgc.m_z + edge.z();
        }
        break;

    case 1:
        if (edge.y() == 0) {
            boundaryIndices[0] = cgc.m_xyz;
        } else {
            boundaryIndices[0] =
                edge.x() * cgc.m_yz + (edge.y() - 1) * cgc.m_z + edge.z();
        }
        if (edge.y() == cgc.m_y) {
            boundaryIndices[1] = cgc.m_xyz;
        } else {
            boundaryIndices[1] =
                edge.x() * cgc.m_yz + edge.y() * cgc.m_z + edge.z();
        }
        break;

    case 2:
        if (edge.z() == 0) {
            boundaryIndices[0] = cgc.m_xyz;
        } else {
            boundaryIndices[0] =
                edge.x() * cgc.m_yz + edge.y() * cgc.m_z + edge.z() - 1;
        }
        if (edge.z() == cgc.m_z) {
            boundaryIndices[1] = cgc.m_xyz;
        } else {
            boundaryIndices[1] =
                edge.x() * cgc.m_yz + edge.y() * cgc.m_z + edge.z();
        }
        break;
    }
    if (kept != nullptr) {
        // Sparse: exterior boundary -> compact sentinel. Masked top-cells have
        // no compact id and resolve to their pocket's virtual node (which
        // makes out-of-grid and masked one lookup).
        for (int i = 0; i < 2; ++i) {
            if (boundaryIndices[i] == static_cast<index_t>(cgc.m_xyz)) {
                boundaryIndices[i] = sentinelIdx;
            } else {
                const int32_t compact =
                    kept->topCellCompact(boundaryIndices[i]);
                if (compact >= 0) {
                    boundaryIndices[i] = static_cast<index_t>(compact);
                } else {
                    // The cube min-corner without an integer division: the two
                    // cofacet cubes of a kept 2-cell are its own coordinate and
                    // the same with the NORMAL axis (== its type) stepped back
                    // by one. i == 0 with that coordinate at 0 took the
                    // sentinel branch above, so the decrement cannot wrap.
                    index_t c[3] = {edge.x(), edge.y(), edge.z()};
                    if (i == 0) {
                        --c[edge.type()];
                    }
                    boundaryIndices[i] =
                        sentinelIdx + static_cast<index_t>(
                                          pockets->ofTopCellAt(c[0], c[1], c[2]));
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
