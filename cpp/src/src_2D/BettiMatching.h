#include "../data_structures.h"
#include "data_structures.h"
#include <memory>
#include <unordered_map>

namespace dim2 {
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
    const vector<vector<VoxelMatch>> &matched = _matched;
    const vector<vector<VoxelPair>> &unmatched0 = _unmatched0;
    const vector<vector<VoxelPair>> &unmatched1 = _unmatched1;
    // Sparse mode only (empty otherwise): birth voxels of the censored
    // ("essential") bars per homology dimension (dim 1 is the top dimension
    // in 2D). Matched lists are row-aligned across inputs; unmatched lists
    // exclude matched classes and are gated per-side by TAU_REAL.
    const vector<vector<vector<index_t>>> &unmatchedEssentials0 =
        _unmatchedEssentials0;
    const vector<vector<vector<index_t>>> &unmatchedEssentials1 =
        _unmatchedEssentials1;
    const vector<vector<vector<index_t>>> &matchedEssentials0 =
        _matchedEssentials0;
    const vector<vector<vector<index_t>>> &matchedEssentials1 =
        _matchedEssentials1;
    const vector<vector<VoxelPair>> &matchedComp = _matchedComp;
    tuple<vector<dim2::RepresentativeCycle>, vector<dim2::RepresentativeCycle>>
    computeRepresentativeCycles(
        const int input, const int dim,
        const optional<vector<size_t>> &matchedPairsIndices,
        const optional<vector<size_t>> &unmatchedPairsIndices);

  private:
    CubicalGridComplex cgc0;
    CubicalGridComplex cgc1;
    CubicalGridComplex cgcComp;
    // Kept-cell index, built once from cgcComp in sparse mode (nullptr in
    // dense mode). Held behind a unique_ptr so its address survives moves of
    // this object (the compact union-finds point at it).
    std::unique_ptr<KeptCells> keptCells;
    // Virtual pocket nodes, built from config.pocketLabels (nullptr in dense
    // mode). Same unique_ptr-for-stable-address reasoning as keptCells: the
    // dual union-finds hold a raw pointer to it.
    std::unique_ptr<PocketNodes> pocketNodes;
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
};
} // namespace dim2
