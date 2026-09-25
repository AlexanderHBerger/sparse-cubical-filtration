#pragma once

#include "BettiMatching.h"
#include "data_structures.h"

#include <unordered_map>

namespace dim3 {
class Dimension0 {
  public:
    // kept: Kept-cell index (sparse mode; nullptr in dense mode) -- the
    // three union-finds become compact over the kept vertices.
    Dimension0(const CubicalGridComplex &cgc0, const CubicalGridComplex &cgc1,
               const CubicalGridComplex &cgcComp, const Config &config,
               vector<Pair> &pairs0, vector<Pair> &pairs1,
               vector<Pair> &pairsComp, vector<Match> &matches,
               unordered_map<uint64_t, bool> &isMatched0,
               unordered_map<uint64_t, bool> &isMatched1,
               unordered_map<uint64_t, size_t> &isMatchedWithIndexComp,
               vector<Cube> &essentials0, vector<Cube> &essentials1,
               vector<pair<size_t, size_t>> &essentialMatches,
               const KeptCells *kept);
    void computePairsAndMatch(vector<Cube> &ctr0, vector<Cube> &ctr1,
                              vector<Cube> &ctrComp);
    void computeInput0Pairs(vector<Cube> &ctr0);
    RepresentativeCycle
    getRepresentativeCycle(const Pair &pair,
                           const CubicalGridComplex &cgc) const;
    tuple<vector<RepresentativeCycle>, vector<RepresentativeCycle>>
    getAllRepresentativeCycles(uint8_t input, bool computeMatchedCycles,
                               bool computeUnmatchedCycles);
    vector<dim3::RepresentativeCycle> computeRepresentativeCycles(
        const int input,
        const std::vector<std::reference_wrapper<Pair>> &requestedPairs);

  private:
    void computePairs(vector<Cube> &ctr, uint8_t k);
    void computeImagePairsAndMatch(vector<Cube> &ctr);
    void enumerateEdges(vector<Cube> &edges,
                        const CubicalGridComplex &cgc) const;
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
    // Essential classes (surviving finite-birth UF roots), sparse mode only;
    // one birth vertex Cube per class.
    vector<Cube> &essentials0;
    vector<Cube> &essentials1;
    // Essential matches as index pairs into (essentials0, essentials1);
    // filled by the essential pass, gated by TAU_REAL.
    vector<pair<size_t, size_t>> &essentialMatches;
    // Injection: input-UF root vertex -> index into essentials
    // vector, for gate-passing (root birth < TAU_REAL) classes only.
    unordered_map<index_t, size_t> essentialMatchMap0;
    unordered_map<index_t, size_t> essentialMatchMap1;
    unordered_map<index_t, Pair> matchMap0;
    unordered_map<index_t, Pair> matchMap1;
    UnionFind uf0;
    UnionFind uf1;
    UnionFind ufComp;
};
} // namespace dim3
