#pragma once

#include "data_structures.h"

namespace dim3 {
class BoundaryEnumerator {
  public:
    Cube nextFace;

    // `kept`/`cone`: sparse mode only. A VIRTUAL column (a cone 2-cell) has a
    // 3-face boundary -- its anchor rim edge plus the two cone edges over the
    // same pocket at the edge's endpoints -- and needs the ConeIndex to look
    // up those endpoints' pocket ranks.
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
    // A cone 2-cell carrying an explicit fundamental cycle has a
    // VARIABLE-length boundary (up to thousands of real edges), so this is
    // no longer a 0..3 slot counter.
    uint32_t position;
    // Resolved once per setBoundaryEnumerator on a virtual cube: its
    // cone2Cell slot and its face count (3 without cycles).
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
