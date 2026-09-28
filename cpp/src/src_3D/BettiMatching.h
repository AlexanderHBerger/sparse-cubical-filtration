#pragma once

#include "../data_structures.h"
#include "data_structures.h"
#include <memory>
#include <unordered_map>

namespace dim3 {
class BettiMatching {
  public:
    BettiMatching(vector<value_t> &&input0, vector<value_t> &&input1,
                  vector<value_t> &&comparison, vector<index_t> &&shape,
                  Config &&config);
    BettiMatching(BettiMatching &&other);
    void computeMatching();
    void computeVoxels();
    vector<vector<VoxelPair>> computePairsInput0();
    void printResult();
    tuple<vector<dim3::RepresentativeCycle>, vector<dim3::RepresentativeCycle>>
    computeRepresentativeCycles(
        const int input, const int dim,
        const optional<vector<size_t>> &matchedPairsIndices,
        const optional<vector<size_t>> &unmatchedPairsIndices);

    const vector<vector<VoxelMatch>> &matched = _matched;
    const vector<vector<VoxelPair>> &unmatched0 = _unmatched0;
    const vector<vector<VoxelPair>> &unmatched1 = _unmatched1;
    const vector<vector<vector<index_t>>> &unmatchedEssentials0 =
        _unmatchedEssentials0;
    const vector<vector<vector<index_t>>> &unmatchedEssentials1 =
        _unmatchedEssentials1;
    const vector<vector<vector<index_t>>> &matchedEssentials0 =
        _matchedEssentials0;
    const vector<vector<vector<index_t>>> &matchedEssentials1 =
        _matchedEssentials1;
    const vector<vector<VoxelPair>> &matchedComp = _matchedComp;

  private:
    CubicalGridComplex cgc0;
    CubicalGridComplex cgc1;
    CubicalGridComplex cgcComp;
    std::unique_ptr<KeptCells> keptCells;
    std::unique_ptr<PocketNodes> pocketNodes;
    std::unique_ptr<ConeIndex> coneIndex;
    vector<vector<Pair>> pairs0;
    vector<vector<Pair>> pairs1;
    vector<vector<Pair>> pairsComp;
    vector<vector<Match>> matches;
    vector<unordered_map<uint64_t, bool>> isMatched0;
    vector<unordered_map<uint64_t, bool>> isMatched1;
    vector<unordered_map<uint64_t, size_t>> isMatchedWithIndexComp;
    vector<vector<Cube>> essentials0;
    vector<vector<Cube>> essentials1;
    vector<vector<pair<size_t, size_t>>> essentialMatches;
    vector<vector<VoxelMatch>> _matched;
    vector<vector<VoxelPair>> _unmatched0;
    vector<vector<VoxelPair>> _unmatched1;
    vector<vector<vector<index_t>>> _unmatchedEssentials0;
    vector<vector<vector<index_t>>> _unmatchedEssentials1;
    vector<vector<vector<index_t>>> _matchedEssentials0;
    vector<vector<vector<index_t>>> _matchedEssentials1;
    vector<vector<VoxelPair>> _matchedComp;
    Config config;
#ifdef USE_CACHE
    optional<SparseOrDenseCubeMap<2, vector<Cube>>> dim1CacheInputPairs0;
    optional<SparseOrDenseCubeMap<2, vector<Cube>>> dim1CacheInputPairs1;
    optional<SparseOrDenseCubeMap<2, vector<Cube>>> dim1CacheCompPairs;
#endif
};
} // namespace dim3
