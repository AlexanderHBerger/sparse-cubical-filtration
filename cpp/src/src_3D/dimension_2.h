#pragma once

#include "../config.h"
#include "BettiMatching.h"
#include "data_structures.h"
#include "enumerators.h"

#include <unordered_map>

namespace dim3 {
class Dimension2 {
  public:
    // kept: Kept-cell index (sparse mode; nullptr in dense mode) -- the
    // three dual union-finds become compact over the kept top-cells and the
    // dual-edge enumerations iterate the kept 2-cell list.
    Dimension2(const CubicalGridComplex &cgc0, const CubicalGridComplex &cgc1,
               const CubicalGridComplex &cgcComp, const Config &config,
               vector<Pair> &pairs0, vector<Pair> &pairs1,
               vector<Pair> &pairsComp, vector<Match> &matches,
               unordered_map<uint64_t, bool> &isMatched0,
               unordered_map<uint64_t, bool> &isMatched1,
               unordered_map<uint64_t, size_t> &isMatchedWithIndexComp,
               vector<Cube> &essentials0, vector<Cube> &essentials1,
               vector<pair<size_t, size_t>> &essentialMatches,
               const KeptCells *kept,
               const PocketNodes *pockets = nullptr);
    void computePairsAndMatch(vector<Cube> &ctr0, vector<Cube> &ctr1,
                              vector<Cube> &ctrComp, vector<Cube> &ctrImage);
    void computeInput0Pairs(vector<Cube> &ctr0);
    RepresentativeCycle
    computeRepresentativeCycle(const Pair &pair,
                               const CubicalGridComplex &cgc) const;
    vector<dim3::RepresentativeCycle> computeRepresentativeCycles(
        const int input,
        const std::vector<std::reference_wrapper<Pair>> &requestedPairs);

  private:
    void enumerateDualEdges(vector<Cube> &dualEdges,
                            const CubicalGridComplex &cgc) const;
    void enumerateDualEdgesComp(vector<Cube> &dualEdges) const;
    void computeInputAndImagePairs(vector<Cube> &dualEdges, const uint8_t &k);
    void computeCompPairsAndMatch(vector<Cube> &dualEdges,
                                  vector<Cube> &ctrImage);
#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP)
    bool isApparentPair(const Cube &dualEdge, BoundaryEnumerator &enumerator,
                        CoboundaryEnumerator &coEnumerator) const;
#endif
    const CubicalGridComplex &cgc0;
    const CubicalGridComplex &cgc1;
    const CubicalGridComplex &cgcComp;
    const Config &config;
    vector<Pair> &pairs0;
    vector<Pair> &pairs1;
    vector<Pair> &pairsComp;
    vector<Match> &matches;
    unordered_map<uint64_t, bool> &isMatched0;
    unordered_map<uint64_t, bool> &isMatched1;
    unordered_map<uint64_t, size_t> &isMatchedWithIndexComp;
    // Essential classes (surviving finite-birth dual roots, excluding the
    // exterior sentinel and masked nodes), sparse mode only. The stored Cube
    // is the dual root top-cell -- the only representative a never-merging
    // dual component has. Censored bars (below) are stored by BIRTH cell.
    vector<Cube> &essentials0;
    vector<Cube> &essentials1;
    // Essential matches as index pairs into (essentials0, essentials1), keyed
    // on COMP dual roots.
    vector<pair<size_t, size_t>> &essentialMatches;
    // Censored bars (dying root is a virtual pocket, so the death is
    // CONE_BIRTH and there is no death voxel) are recorded in essentials0/1 by
    // their BIRTH cell and joined through these maps, keyed by the
    // comparison-side dying root exactly as matchMap0/1 are.
    unordered_map<index_t, size_t> censoredMatchMap0;
    unordered_map<index_t, size_t> censoredMatchMap1;
    unordered_map<index_t, size_t> essentialMatchMap0;
    unordered_map<index_t, size_t> essentialMatchMap1;
    unordered_map<index_t, Pair> matchMap0;
    unordered_map<index_t, Pair> matchMap1;
    // Kept-cell index (sparse mode only; nullptr in dense mode).
    const KeptCells *kept;
    const PocketNodes *pockets;
    UnionFindDual uf0;
    UnionFindDual uf1;
    UnionFindDual ufComp;
};
} // namespace dim3
