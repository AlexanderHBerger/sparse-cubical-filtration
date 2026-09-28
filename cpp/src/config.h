#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// #define RUNTIME
#define COMPUTE_COMPARISON
// #define USE_ISPAIRED
// #define USE_REDUCTION_MATRIX
// #define USE_APPARENT_PAIRS_COMP
// #define USE_APPARENT_PAIRS
#define USE_EMERGENT_PAIRS
#define USE_CACHE
// #define USE_CLEARING_IMAGE
#define USE_CLEARING_DIM0
#define USE_STABLE_SORT_OR_STABLE_PARTITION // enables stable sort/binary input
                                            // sorting optimization in dim3 edge
                                            // enumeration methods
#ifndef NO_PARALLELIZE_DIM1
#define PARALLELIZE_INDEPENDENT_BARCODES_DIM1 // enables parallelization of
                                              // independent barcode
                                              // computations in
                                              // dim3::Dimension1
#endif

typedef uint32_t index_t;
#ifdef VALUE_T_FLOAT
typedef float value_t;
#else
typedef double value_t;
#endif

#define INFTY numeric_limits<value_t>::infinity()
#define NONE numeric_limits<index_t>::max()
#define NONE_INDEX numeric_limits<uint64_t>::max()

enum fileFormat { DIPHA, PERSEUS, NUMPY };

using namespace std;

#if defined(PARALLELIZE_INDEPENDENT_BARCODES_DIM1) and                         \
    defined(USE_CLEARING_IMAGE)
static_assert(false,
              "PARALLELIZE_INDEPENDENT_BARCODES_DIM1 is active alongside "
              "incompatible options (race conditions could occur!)");
#endif

struct Config {
    value_t threshold = INFTY;

    size_t minRecursionToCache = 1;
    size_t cacheSize = numeric_limits<size_t>::max();

    // Sparse cubical complex: voxels with min(input0, input1) >= maskThreshold
    // are set to INFTY in both inputs by the caller, and every connected
    // component of these masked voxels (a pocket) is replaced by one virtual
    // node born at CONE_BIRTH.
    bool sparseComplex = false;
    value_t maskThreshold = INFTY;
    bool releaseCachesAfterMatching = false;

    // 0 on kept voxels, 1..P on masked voxels (26-connected components, 8 in
    // 2D), numbered in raster order of first occurrence.
    std::shared_ptr<const std::vector<int32_t>> pocketLabels;
    // Emit only a minimal generating set of the dim-1 cone columns.
    bool conePrune = true;
};

// Background value of g = 1 - p. All virtual cells of the sparse complex are
// born here, after every kept cell of equal value.
constexpr value_t TAU_REAL = 1.0;
constexpr value_t CONE_BIRTH = TAU_REAL;
