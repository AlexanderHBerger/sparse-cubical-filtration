#pragma once

#include "BettiMatching.h"
#include "data_structures.h"
#include "enumerators.h"

#include <queue>
#include <unordered_map>

namespace dim3 {
typedef priority_queue<Cube, vector<Cube>, CubeComparator> CubeQueue;

class Dimension1 {
  public:
    Dimension1(const CubicalGridComplex &cgc0, const CubicalGridComplex &cgc1,
               const CubicalGridComplex &cgcComp, const Config &config,
               vector<Pair> &pairs0, vector<Pair> &pairs1,
               vector<Pair> &pairsComp, vector<Match> &matches,
               unordered_map<uint64_t, bool> &isMatched0,
               unordered_map<uint64_t, bool> &isMatched1,
               unordered_map<uint64_t, size_t> &isMatchedWithIndexComp,
               vector<Cube> &essentials0, vector<Cube> &essentials1,
               vector<pair<size_t, size_t>> &essentialMatches,
               const KeptCells *kept,
               const ConeIndex *cone
#ifdef USE_CACHE
               ,
               SparseOrDenseCubeMap<2, vector<Cube>> &_cacheInputPairs0,
               SparseOrDenseCubeMap<2, vector<Cube>> &_cacheInputPairs1,
               SparseOrDenseCubeMap<2, vector<Cube>> &_cacheCompPairs
#endif
    );
    void computePairsAndMatch(vector<Cube> &ctr0, vector<Cube> &ctr1,
                              vector<Cube> &ctrComp, vector<Cube> &ctrImage);
    void computeInput0Pairs(vector<Cube> &ctr0);
    vector<dim3::RepresentativeCycle> computeRepresentativeCycles(
        const int input,
        const std::vector<std::reference_wrapper<Pair>> &requestedPairs);

  private:
    enum ComputePairsMode { INPUT_PAIRS, COMPARISON_PAIRS, IMAGE_PAIRS };

    void computePairs(vector<Cube> &ctr, uint8_t k);
    void computePairsComp(vector<Cube> &ctr);
    void computePairsImage(vector<Cube> &ctr, uint8_t k);

    template <ComputePairsMode computePairsMode>
    void computePairsUnified(vector<Cube> &ctr, uint8_t k
#ifdef USE_CACHE
                             ,
                             SparseOrDenseCubeMap<2, vector<Cube>> &cache
#endif
    );

    void computeMatching();
    void appendConeColumns(vector<Cube> &ctr,
                           const CubicalGridComplex &cgc) const;
    void
    enumerateEdges(vector<Cube> &edges, const CubicalGridComplex &cgc,
                   SparseOrDenseCubeMap<1, size_t> &pivotColumnIndex) const;
    void enumerateColumnsToReduce(vector<Cube> &ctr,
                                  const CubicalGridComplex &cgc) const;
    Cube popPivot(CubeQueue &column) const;
    Cube getPivot(CubeQueue &column) const;
#ifdef USE_REDUCTION_MATRIX
    void useReductionMatrix(const Cube &column, CubeQueue &workingBoundary,
                            BoundaryEnumerator &enumerator
#ifdef USE_CACHE
                            ,
                            SparseOrDenseCubeMap<2, vector<Cube>> &cache
#endif
    ) const;
#endif
#ifdef USE_CACHE
    bool columnIsCached(const Cube &column, CubeQueue &workingBoundary,
                        SparseOrDenseCubeMap<2, vector<Cube>> &cache) const;
    void addCache(const Cube &column, CubeQueue &workingBoundary,
                  queue<uint64_t> &cachedColumnIdx,
                  SparseOrDenseCubeMap<2, vector<Cube>> &cache);
#endif
#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP)
    bool pivotIsApparentPair(const Cube &pivot, vector<Cube> &faces,
                             BoundaryEnumerator &enumerator,
                             CoboundaryEnumerator &coEnumerator) const;
    bool pivotOfColumnIsApparentPair(const Cube &pivot, const Cube &column,
                                     vector<Cube> &faces,
                                     BoundaryEnumerator &enumerator,
                                     CoboundaryEnumerator &coEnumerator) const;
#endif
#ifdef USE_EMERGENT_PAIRS
    template <ComputePairsMode computePairsMode>
    bool
    isEmergentPair(const Cube &column, Cube &pivot, size_t &j,
                   vector<Cube> &faces, bool &checkEmergentPair,
                   const CubicalGridComplex &cgc,
                   BoundaryEnumerator &enumerator,
                   BoundaryEnumerator &enumeratorAP,
                   CoboundaryEnumerator &coEnumeratorAP,
                   SparseOrDenseCubeMap<1, size_t> &pivotColumnIndex) const;
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
    const KeptCells *kept;
#ifdef USE_ISPAIRED
    unordered_map<uint64_t, bool> isPairedComp;
#endif
    SparseOrDenseCubeMap<1, Pair> matchMap0;
    SparseOrDenseCubeMap<1, Pair> matchMap1;
    SparseOrDenseCubeMap<2, uint64_t> matchMapIm0;
    SparseOrDenseCubeMap<2, uint64_t> matchMapIm1;

    const ConeIndex *cone;
    vector<Cube> &essentials0;
    vector<Cube> &essentials1;
    vector<pair<size_t, size_t>> &essentialMatches;
    unordered_map<uint64_t, size_t> censoredIndex0;
    unordered_map<uint64_t, size_t> censoredIndex1;

    // to be used for input 0 pairs
    SparseOrDenseCubeMap<1, size_t> pivotColumnIndexInput0;
    // to be used for input 1 pairs
    SparseOrDenseCubeMap<1, size_t> pivotColumnIndexInput1;
    // to be used for comparison pairs
    SparseOrDenseCubeMap<1, size_t> pivotColumnIndexComp;
    // to be used for image pairs 0
    SparseOrDenseCubeMap<1, size_t> pivotColumnIndexImage0;
    // to be used for image pairs 1
    SparseOrDenseCubeMap<1, size_t> pivotColumnIndexImage1;
#ifdef USE_REDUCTION_MATRIX
    SparseOrDenseCubeMap<2, vector<Cube>> reductionMatrix;
#endif

#ifdef USE_CACHE
    // Keep the caches for input 0 and input 1 barcodes for later use in
    // computing representative cycles.
    SparseOrDenseCubeMap<2, vector<Cube>> &cacheInputPairs0;
    SparseOrDenseCubeMap<2, vector<Cube>> &cacheInputPairs1;
    SparseOrDenseCubeMap<2, vector<Cube>> &cacheCompPairs;
#endif
};
} // namespace dim3
