#include "enumerators.h"

#include <iostream>
using namespace std;

using namespace dim3;

BoundaryEnumerator::BoundaryEnumerator(const CubicalGridComplex &_cgc,
                                       const KeptCells *_kept,
                                       const ConeIndex *_cone,
                                       value_t _pocketBirth)
    : cgc(_cgc), kept(_kept), cone(_cone), pocketBirth(_pocketBirth) {
    nextFace = Cube();
}

void BoundaryEnumerator::setBoundaryEnumerator(const Cube &_cube) {
    cube = _cube;
    position = 0;
    if (cone != nullptr && isVirtualCell(_cube.index)) {
        if (cone->hasCycles()) {
            // Resolve the column's cone2Cell slot ONCE, not per face.
            const uint8_t code = _cube.type() - VIRTUAL_TYPE_BASE;
            const uint8_t anchorType = code / 4;
            const uint8_t rank = code % 4;
            const uint64_t anchorKey = (_cube.index & ~(uint64_t)0xf) |
                                       (uint64_t)anchorType;
            const int32_t e = kept->edgeCompact(anchorKey);
            coneSlot = cone->cone2CellSlot(e, rank);
            coneFaces = static_cast<uint32_t>(cone->cycleLength(coneSlot));
        } else {
            coneSlot = -1;
            coneFaces = 3;
        }
    }
}

// Boundary of a virtual (cone) 2-cell, in DECREASING order.
//
// With explicit fundamental cycles (cone_prune=true) it is the stored cycle --
// real kept edges only, variable length -- and `slot` indexes it directly,
// because the CSR is written in decreasing Cube-index order. Index order,
// which is complex-independent, is enough: every consumer but one pushes into
// a priority queue, and the one exception, `isEmergentPair`, looks only at
// faces whose birth EQUALS the column's, of which a cycle column has none --
// its faces are real kept edges, at most CONE_BIRTH. A face AT CONE_BIRTH (a
// kept edge at the background value) can tie with the column, and then the
// check fires and produces a zero-persistence (CONE_BIRTH, CONE_BIRTH) pair,
// which is dropped -- a cycle born at background dying at background, exactly
// as in the dense filtration.
// (birth, index) order is not needed and would differ between cgc0, cgc1 and
// cgcComp.
//
// Without them (cone_prune=false) the boundary is the 3-face
// cone2Cell(p, e) = e + coneEdge(p, v0) + coneEdge(p, v1).
// `slot` counts down the DECREASING (birth, index) order that the emergent-pair
// lemma relies on: both cone edges are born at CONE_BIRTH, at or above the
// anchor edge, and at equal birth their virtual type (>= 3) outranks the
// anchor's; coneEdge(p,v1) outranks coneEdge(p,v0) because v1 = v0 + e_axis
// sorts higher. So the order is always v1, v0, e.
bool BoundaryEnumerator::virtualFace(uint32_t slot) {
    if (slot >= coneFaces) {
        return false;
    }
    if (coneSlot >= 0) {
        // Explicit fundamental cycle: real kept edges only, already stored in
        // DECREASING Cube-index order, so `slot` counts down the order the
        // emergent-pair lemma needs (see the comment above virtualFace).
        const uint64_t k = cone->cycleEdge(coneSlot, static_cast<int32_t>(slot));
        const index_t ex = (k >> 44) & 0xfffff;
        const index_t ey = (k >> 24) & 0xfffff;
        const index_t ez = (k >> 4) & 0xfffff;
        const uint8_t et = k & 0xf;
        nextFace = Cube(cgc.getBirth(ex, ey, ez, et, 1), ex, ey, ez, et);
        return true;
    }
    const index_t x = cube.x(), y = cube.y(), z = cube.z();
    const uint8_t code = cube.type() - VIRTUAL_TYPE_BASE;
    const uint8_t anchorType = code / 4;
    const uint8_t rank = code % 4;
    if (slot == 2) {
        nextFace = Cube(cgc.getBirth(x, y, z, anchorType, 1), x, y, z,
                        anchorType);
        return true;
    }
    const uint64_t anchorKey = ((uint64_t)x << 44) | ((uint64_t)y << 24) |
                               ((uint64_t)z << 4) | (uint64_t)anchorType;
    const int32_t e = kept->edgeCompact(anchorKey);
    index_t v[3] = {x, y, z};
    uint8_t vRank;
    if (slot == 0) {
        ++v[anchorType];
        vRank = cone->rankAtV1(e, rank);
    } else {
        vRank = cone->rankAtV0(e, rank);
    }
    // V-construction rule with the apex treated as a virtual voxel at
    // CONE_BIRTH: a cone edge's birth is max(CONE_BIRTH, its real vertex).
    // BettiMatching has checked no kept voxel exceeds CONE_BIRTH, so that is
    // CONE_BIRTH -- identical in cgc0, cgc1 and cgcComp, which is what the
    // shared virtual slice needs.
    nextFace = Cube(pocketBirth, v[0], v[1], v[2],
                    VIRTUAL_TYPE_BASE + vRank);
    return true;
}

bool BoundaryEnumerator::hasPreviousFace() {
    if (cone != nullptr && isVirtualCell(cube.index)) {
        return virtualFace(position++);
    }
    if (position == 4) {
        return false;
    } else {
        index_t x = cube.x();
        index_t y = cube.y();
        index_t z = cube.z();
        value_t birth;
        switch (cube.type()) {
        case 0:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x, y + 1, z, 2, 1);
                nextFace = Cube(birth, x, y + 1, z, 2);
                break;

            case 1:
                birth = cgc.getBirth(x, y, z + 1, 1, 1);
                nextFace = Cube(birth, x, y, z + 1, 1);
                break;

            case 2:
                birth = cgc.getBirth(x, y, z, 2, 1);
                nextFace = Cube(birth, x, y, z, 2);
                break;

            case 3:
                birth = cgc.getBirth(x, y, z, 1, 1);
                nextFace = Cube(birth, x, y, z, 1);
                break;
            }
            break;

        case 1:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x + 1, y, z, 2, 1);
                nextFace = Cube(birth, x + 1, y, z, 2);
                break;

            case 1:
                birth = cgc.getBirth(x, y, z + 1, 0, 1);
                nextFace = Cube(birth, x, y, z + 1, 0);
                break;

            case 2:
                birth = cgc.getBirth(x, y, z, 2, 1);
                nextFace = Cube(birth, x, y, z, 2);
                break;

            case 3:
                birth = cgc.getBirth(x, y, z, 0, 1);
                nextFace = Cube(birth, x, y, z, 0);
                break;
            }
            break;

        case 2:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x + 1, y, z, 1, 1);
                nextFace = Cube(birth, x + 1, y, z, 1);
                break;

            case 1:
                birth = cgc.getBirth(x, y + 1, z, 0, 1);
                nextFace = Cube(birth, x, y + 1, z, 0);
                break;

            case 2:
                birth = cgc.getBirth(x, y, z, 1, 1);
                nextFace = Cube(birth, x, y, z, 1);
                break;

            case 3:
                birth = cgc.getBirth(x, y, z, 0, 1);
                nextFace = Cube(birth, x, y, z, 0);
                break;
            }
            break;
        }
        ++position;
        return true;
    }
}

bool BoundaryEnumerator::hasNextFace() {
    if (cone != nullptr && isVirtualCell(cube.index)) {
        // The reverse of hasPreviousFace's order.
        if (position >= coneFaces) {
            return false;
        }
        return virtualFace(coneFaces - 1 - position++);
    }
    if (position == 4) {
        return false;
    } else {
        index_t x = cube.x();
        index_t y = cube.y();
        index_t z = cube.z();
        value_t birth;
        switch (cube.type()) {
        case 0:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x, y, z, 1, 1);
                nextFace = Cube(birth, x, y, z, 1);
                break;

            case 1:
                birth = cgc.getBirth(x, y, z, 2, 1);
                nextFace = Cube(birth, x, y, z, 2);
                break;

            case 2:
                birth = cgc.getBirth(x, y, z + 1, 1, 1);
                nextFace = Cube(birth, x, y, z + 1, 1);
                break;

            case 3:
                birth = cgc.getBirth(x, y + 1, z, 2, 1);
                nextFace = Cube(birth, x, y + 1, z, 2);
                break;
            }
            break;

        case 1:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x, y, z, 0, 1);
                nextFace = Cube(birth, x, y, z, 0);
                break;

            case 1:
                birth = cgc.getBirth(x, y, z, 2, 1);
                nextFace = Cube(birth, x, y, z, 2);
                break;

            case 2:
                birth = cgc.getBirth(x, y, z + 1, 0, 1);
                nextFace = Cube(birth, x, y, z + 1, 0);
                break;

            case 3:
                birth = cgc.getBirth(x + 1, y, z, 2, 1);
                nextFace = Cube(birth, x + 1, y, z, 2);
                break;
            }
            break;

        case 2:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x, y, z, 0, 1);
                nextFace = Cube(birth, x, y, z, 0);
                break;

            case 1:
                birth = cgc.getBirth(x, y, z, 1, 1);
                nextFace = Cube(birth, x, y, z, 1);
                break;

            case 2:
                birth = cgc.getBirth(x, y + 1, z, 0, 1);
                nextFace = Cube(birth, x, y + 1, z, 0);
                break;

            case 3:
                birth = cgc.getBirth(x + 1, y, z, 1, 1);
                nextFace = Cube(birth, x + 1, y, z, 1);
                break;
            }
            break;
        }
        ++position;
        return true;
    }
}

CoboundaryEnumerator::CoboundaryEnumerator(const CubicalGridComplex &_cgc)
    : cgc(_cgc) {
    nextCoface = Cube();
}

void CoboundaryEnumerator::setCoboundaryEnumerator(const Cube &_cube) {
    cube = _cube;
    position = 0;
}

bool CoboundaryEnumerator::hasNextCoface() {
    if (position == 4) {
        return false;
    } else {
        index_t x = cube.x();
        index_t y = cube.y();
        index_t z = cube.z();
        value_t birth;
        switch (cube.type()) {
        case 0:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x, y - 1, z, 2, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y - 1, z, 2);
                    break;
                } else {
                    ++position;
                }

            case 1:
                birth = cgc.getBirth(x, y, z - 1, 1, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z - 1, 1);
                    break;
                } else {
                    ++position;
                }

            case 2:
                birth = cgc.getBirth(x, y, z, 1, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z, 1);
                    break;
                } else {
                    ++position;
                }

            case 3:
                birth = cgc.getBirth(x, y, z, 2, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z, 2);
                    break;
                } else {
                    ++position;
                }

            case 4:
                return false;
            }
            break;

        case 1:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x - 1, y, z, 2, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x - 1, y, z, 2);
                    break;
                } else {
                    ++position;
                }

            case 1:
                birth = cgc.getBirth(x, y, z - 1, 0, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z - 1, 0);
                    break;
                } else {
                    ++position;
                }

            case 2:
                birth = cgc.getBirth(x, y, z, 0, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z, 0);
                    break;
                } else {
                    ++position;
                }

            case 3:
                birth = cgc.getBirth(x, y, z, 2, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z, 2);
                    break;
                } else {
                    ++position;
                }

            case 4:
                return false;
            }
            break;

        case 2:
            switch (position) {
            case 0:
                birth = cgc.getBirth(x - 1, y, z, 1, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x - 1, y, z, 1);
                    break;
                } else {
                    ++position;
                }

            case 1:
                birth = cgc.getBirth(x, y - 1, z, 0, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y - 1, z, 0);
                    break;
                } else {
                    ++position;
                }

            case 2:
                birth = cgc.getBirth(x, y, z, 0, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z, 0);
                    break;
                } else {
                    ++position;
                }

            case 3:
                birth = cgc.getBirth(x, y, z, 1, 2);
                if (birth != INFTY) {
                    nextCoface = Cube(birth, x, y, z, 1);
                    break;
                } else {
                    ++position;
                }

            case 4:
                return false;
            }
            break;
        }
        ++position;
        return true;
    }
}
