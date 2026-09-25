#pragma once

#include "data_structures.h"

#include <unordered_map>

namespace dim2 {
class Dimension1 {
  public:
    // kept: Kept-cell index (sparse mode; nullptr in dense mode) -- the
    // three dual union-finds become compact over the kept top-cells and the
    // dual-edge enumerations iterate the kept edge list.
    Dimension1(const CubicalGridComplex &cgc0, const CubicalGridComplex &cgc1,
               const CubicalGridComplex &cgcComp, const Config &config,
               vector<Pair> &pairs0, vector<Pair> &pairs1,
               vector<Pair> &pairsComp, vector<Match> &matches,
               unordered_map<uint64_t, bool> &isMatched0,
               unordered_map<uint64_t, bool> &isMatched1,
               unordered_map<uint64_t, size_t> &isMatchedWithIndexComp,
               vector<Cube> &essentials0, vector<Cube> &essentials1,
               vector<pair<size_t, size_t>> &essentialMatches,
               const KeptCells *kept, const PocketNodes *pockets = nullptr);
    void computePairsAndMatch(vector<Cube> &ctr0, vector<Cube> &ctr1,
                              vector<Cube> &ctrComp);
    void computeInput0Pairs(vector<Cube> &ctr0);
    vector<dim2::RepresentativeCycle> computeRepresentativeCycles(
        const int input,
        const std::vector<std::reference_wrapper<Pair>> &requestedPairs);

  private:
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
    // exterior sentinel and masked nodes), sparse mode only; the stored Cube
    // is the dual root top-cell; censored bars (below) are stored by BIRTH
    // cell.
    vector<Cube> &essentials0;
    vector<Cube> &essentials1;
    // Essential matches, comp-root keyed; mirrors
    // src_3D/dimension_2.
    vector<pair<size_t, size_t>> &essentialMatches;
    unordered_map<index_t, size_t> essentialMatchMap0;
    unordered_map<index_t, size_t> essentialMatchMap1;
    // Censored bars (dying root is a virtual pocket node, so the
    // death is CONE_BIRTH and there is no death voxel). They are recorded in
    // essentials0/1 by their BIRTH cell and joined through these maps, keyed
    // by the comparison-side dying root exactly as matchMap0/1 are.
    unordered_map<index_t, size_t> censoredMatchMap0;
    unordered_map<index_t, size_t> censoredMatchMap1;
    unordered_map<index_t, Pair> matchMap0;
    unordered_map<index_t, Pair> matchMap1;
    // Kept-cell index (sparse mode only; nullptr in dense mode).
    const KeptCells *kept;
    // Virtual pocket nodes (nullptr in dense mode).
    const PocketNodes *pockets;
    UnionFindDual uf0;
    UnionFindDual uf1;
    UnionFindDual ufComp;

    void enumerateDualEdges(vector<Cube> &dualEdges,
                            const CubicalGridComplex &cgc) const;
    void computeInputAndImagePairs(vector<Cube> &dualEdges, const uint8_t &k);
    void computeCompPairsAndMatch(vector<Cube> &dualEdges);
};
} // namespace dim2
