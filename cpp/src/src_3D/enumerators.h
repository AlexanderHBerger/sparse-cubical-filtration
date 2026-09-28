#pragma once

#include "data_structures.h"

namespace dim3 {
class BoundaryEnumerator {
  public:
    Cube nextFace;

    BoundaryEnumerator(const CubicalGridComplex &cgc,
                       const KeptCells *kept = nullptr,
                       const ConeIndex *cone = nullptr,
                       value_t pocketBirth = INFTY);
    void setBoundaryEnumerator(const Cube &cube);
    bool hasPreviousFace();
    bool hasNextFace();

  private:
    bool virtualFace(uint32_t slot);

    const CubicalGridComplex &cgc;
    const KeptCells *kept;
    const ConeIndex *cone;
    value_t pocketBirth;
    Cube cube;
    uint32_t position;
    int32_t coneSlot = -1;
    uint32_t coneFaces = 0;
};

class CoboundaryEnumerator {
  public:
    Cube nextCoface;

    CoboundaryEnumerator(const CubicalGridComplex &cgc);
    void setCoboundaryEnumerator(const Cube &cube);
    bool hasNextCoface();

  private:
    const CubicalGridComplex &cgc;
    Cube cube;
    uint8_t position;
};
} // namespace dim3
