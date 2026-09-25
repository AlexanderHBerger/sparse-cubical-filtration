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
// Build with -DNO_PARALLELIZE_DIM1 to run the five dim-1 reductions
// sequentially. Only useful for PROFILING: under -DRUNTIME the five reductions
// print concurrently and their labels interleave with their numbers, so per
// reduction attribution is guesswork. Sequential mode makes the output
// unambiguous. Default (flag unset) is unchanged.
#ifndef NO_PARALLELIZE_DIM1
#define PARALLELIZE_INDEPENDENT_BARCODES_DIM1 // enables parallelization of
                                              // independent barcode
                                              // computations in
                                              // dim3::Dimension1
#endif

typedef uint32_t index_t;
// value_t is the filtration-value type. The PH core only compares/max/min
// these (never accumulates), so float is numerically safe and halves the
// value-grid memory + bandwidth. Build with -DVALUE_T_FLOAT for the fp32
// variant (results are equivalent-within-tolerance to the fp64 build, not
// bit-identical). Default stays double for backward compatibility.
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

    // Sparse (pocket-cone) Betti matching. All dense code paths are
    // unchanged when sparseComplex == false (the default).
    //
    // The caller masks both inputs identically: a voxel is kept iff
    // min(input0, input1) < maskThreshold, and every other voxel is set to
    // INFTY in both inputs. The kept voxels span the kept complex K. Every
    // connected component of masked voxels (a "pocket") is replaced by ONE
    // virtual node born at CONE_BIRTH, and in 3D the dimension-1 reduction
    // additionally carries the cone over each pocket's rim. The resulting
    // barcode is the persistent homology of dense(where(keep, g, CONE_BIRTH)):
    // below CONE_BIRTH it is the persistent homology of K at the inputs' own
    // values, and every class that survives in K is censored at CONE_BIRTH.
    bool sparseComplex = false;
    value_t maskThreshold = INFTY;
    // Free the dim-1 reduction caches right after matching instead of
    // retaining them for computeRepresentativeCycles. Valid for dense and
    // sparse; changes memory lifetime only, never results.
    bool releaseCachesAfterMatching = false;

    // Voxel-level pocket labels: 0 on kept voxels, 1..P on masked ones, one
    // label per 26-connected component of the masked voxels (8-connected in
    // 2D), numbered in raster order of first occurrence (see
    // sparse_bm/pocket_labels.py).
    //
    // 26-connectivity is required: two cubes merge in the dual sweep through
    // their shared 2-cell, at that 2-cell's value, so the dual components of
    // the masked region are the 26-connected components of the masked voxels,
    // not the 6-connected components of the non-kept cubes (the coarser rule
    // merges cavities that are separated by a kept 2-cell).
    //
    // shared_ptr because Config is copied by value into the per-sample
    // std::async tasks in _BettiMatching.cpp.
    std::shared_ptr<const std::vector<int32_t>> pocketLabels;
    // Emit only a minimal generating set of cone 2-cell columns instead of one
    // per (rim edge, pocket). The dropped columns are linear combinations of
    // rim 2-cells (real columns of the matrix) and sibling cone columns, all
    // of which lie in K and are therefore born no later than the dropped
    // column (the whole of K precedes the cone, which sits at CONE_BIRTH). The
    // barcode is unchanged; only the number of expensive columns drops.
    bool conePrune = true;
};

// The BACKGROUND value of the fg-low [0, 1] convention (g = 1 - p). Two roles,
// one number:
//
//   TAU_REAL   the essential-matching gate: a class born at the background
//              value is not a feature.
//   CONE_BIRTH the value of every VIRTUAL cell of the sparse complex -- cone
//              edges, cone 2-cells and the dual pocket nodes. The masked region
//              is background in BOTH inputs (min(I, J) >= tau, and for a binary
//              target that means exactly 1), so it enters at the background
//              value in every complex: the target's sparse filtration is then
//              literally its dense one (where(keep, g_t, 1) == g_t), and the
//              prediction's is where(keep, g_p, 1). BettiMatching throws if a
//              kept voxel exceeds it.
//
// Consequences:
//
//   * A class born at the background value dies at it -- zero persistence,
//     never a bar, on either side. A prediction feature the target lacks is
//     therefore UNMATCHED, exactly as in the dense matching.
//   * Equal values are ordered KEPT BEFORE VIRTUAL, in every dimension: the
//     dim-1 cone block is appended after the real columns (never merged into
//     them), and the dual union-find's original-index tie-break puts every
//     pocket after every kept top cell. So "all of K precedes the cone" holds
//     in the total order even at the background value, and with it: every
//     FINITE bar is exactly a bar of the masked filtration PH(K, f) (columns
//     appended after every real one cannot disturb a pair already made); the
//     cone-pruning replacement chain, which lives in K, is present when a
//     column is dropped; and the three reductions claim the same rows with
//     their real columns, so a class is censored on one side iff it is
//     censored on the other (computeMatching throws if a mixed censored/finite
//     join ever occurs).
//   * Values above tau keep their real ordering, and the only ties with
//     virtual cells are at the background value.
constexpr value_t TAU_REAL = 1.0;
constexpr value_t CONE_BIRTH = TAU_REAL;
