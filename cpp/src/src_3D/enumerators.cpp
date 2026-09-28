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

// Faces of a cone 2-cell in decreasing order, as the emergent pair check
// expects. With pruning, the face list is the stored fundamental cycle.
// Otherwise the boundary is coneEdge(p, v1), coneEdge(p, v0), e.
bool BoundaryEnumerator::virtualFace(uint32_t slot) {
    if (slot >= coneFaces) {
        return false;
    }
    if (coneSlot >= 0) {
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
