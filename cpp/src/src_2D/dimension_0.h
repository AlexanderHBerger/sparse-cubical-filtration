#pragma once

#include "data_structures.h"

#include <unordered_map>

namespace dim2 {
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
    // Essential classes (surviving finite-birth UF roots), sparse mode only;
    // one birth vertex Cube per class.
    vector<Cube> &essentials0;
    vector<Cube> &essentials1;
    // Essential matches + the injection maps, mirroring
    // src_3D/dimension_0.
    vector<pair<size_t, size_t>> &essentialMatches;
    unordered_map<index_t, size_t> essentialMatchMap0;
    unordered_map<index_t, size_t> essentialMatchMap1;
    unordered_map<index_t, Pair> matchMap0;
    unordered_map<index_t, Pair> matchMap1;
    UnionFind uf0;
    UnionFind uf1;
    UnionFind ufComp;

    void computePairs(vector<Cube> &ctr, uint8_t k);
    void computeImagePairsAndMatch(vector<Cube> &ctr);
    void enumerateEdges(vector<Cube> &edges,
                        const CubicalGridComplex &cgc) const;
};
} // namespace dim2
