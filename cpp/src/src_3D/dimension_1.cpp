#include "dimension_1.h"
#include "BettiMatching.h"
#include "data_structures.h"
#include "enumerators.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace dim3;
using namespace std::chrono;

// Can `ctrImage = ctrComp` be taken BEFORE the comparison reduction instead of
// after it? If so, the image reductions no longer have to wait for the
// comparison one and all five dim-1 reductions start together -- which matters a
// lot, because they are heavily unbalanced (see computePairsAndMatch).
//
// It is safe exactly when nothing the image passes need is produced by the
// comparison pass:
//   USE_CLEARING_IMAGE      -- computePairsComp() then ERASES entries from
//                             ctrComp, so the copy must be taken afterwards.
//   USE_APPARENT_PAIRS_COMP -- ctrImage is built in Dimension2, not copied here.
//   USE_ISPAIRED            -- the image passes read isPairedComp[], which the
//                             comparison pass writes.
// None of the three is enabled by default. Without USE_CLEARING_DIM0 the image
// passes read ctrComp directly and there was never a barrier.
// Build with -DNO_DIM1_CTRIMAGE_HOIST to run the comparison reduction before
// the image reductions (for timing comparisons only).
#if defined(USE_CLEARING_DIM0) and not defined(USE_APPARENT_PAIRS_COMP) and    \
    not defined(USE_CLEARING_IMAGE) and not defined(USE_ISPAIRED) and          \
    not defined(NO_DIM1_CTRIMAGE_HOIST)
#define DIM1_CTRIMAGE_HOISTED
#endif

Dimension1::Dimension1(const CubicalGridComplex &_cgc0,
                       const CubicalGridComplex &_cgc1,
                       const CubicalGridComplex &_cgcComp,
                       const Config &_config, vector<Pair> &_pairs0,
                       vector<Pair> &_pairs1, vector<Pair> &_pairsComp,
                       vector<Match> &_matches,
                       unordered_map<uint64_t, bool> &_isMatched0,
                       unordered_map<uint64_t, bool> &_isMatched1,
                       unordered_map<uint64_t, size_t> &_isMatchedWithIndexComp,
                       vector<Cube> &_essentials0, vector<Cube> &_essentials1,
                       vector<pair<size_t, size_t>> &_essentialMatches,
                       const KeptCells *_kept, const ConeIndex *_cone
#ifdef USE_CACHE
                       ,
                       SparseOrDenseCubeMap<2, vector<Cube>> &_cacheInputPairs0,
                       SparseOrDenseCubeMap<2, vector<Cube>> &_cacheInputPairs1,
                       SparseOrDenseCubeMap<2, vector<Cube>> &_cacheCompPairs
#endif
                       )
    : cgc0(_cgc0), cgc1(_cgc1), cgcComp(_cgcComp), config(_config),
      pairs0(_pairs0), pairs1(_pairs1), pairsComp(_pairsComp),
      matches(_matches), isMatched0(_isMatched0), isMatched1(_isMatched1),
      isMatchedWithIndexComp(_isMatchedWithIndexComp), kept(_kept),
      matchMap0(_cgc0.shape, _kept, _cone),
      matchMap1(_cgc0.shape, _kept, _cone),
      matchMapIm0(_cgc0.shape, _kept, _cone),
      matchMapIm1(_cgc0.shape, _kept, _cone), cone(_cone),
      essentials0(_essentials0), essentials1(_essentials1),
      essentialMatches(_essentialMatches),
#ifdef USE_REDUCTION_MATRIX
      reductionMatrix(_cgc0.shape, _kept),
#endif
      pivotColumnIndexInput0(_cgc0.shape, _kept, _cone),
      pivotColumnIndexInput1(_cgc1.shape, _kept, _cone),
      pivotColumnIndexComp(_cgcComp.shape, _kept, _cone),
      pivotColumnIndexImage0(_cgc0.shape, _kept, _cone),
      pivotColumnIndexImage1(_cgc1.shape, _kept, _cone),
#ifdef USE_CACHE
      cacheInputPairs0(_cacheInputPairs0), cacheInputPairs1(_cacheInputPairs1),
      cacheCompPairs(_cacheCompPairs)
#endif
{
    if (cone != nullptr) {
        // Cone-edge keys are resolved through the dense vertex id space
        // (x*n_yz + y*n_z + z), so every edge-keyed map needs its strides.
        const size_t sy = static_cast<size_t>(_cgc0.shape[1]);
        const size_t sz = static_cast<size_t>(_cgc0.shape[2]);
        for (SparseOrDenseCubeMap<1, size_t> *m :
             {&pivotColumnIndexInput0, &pivotColumnIndexInput1,
              &pivotColumnIndexComp, &pivotColumnIndexImage0,
              &pivotColumnIndexImage1}) {
            m->setVertexStrides(sy * sz, sz);
        }
        matchMap0.setVertexStrides(sy * sz, sz);
        matchMap1.setVertexStrides(sy * sz, sz);
    }
}

void Dimension1::computePairsAndMatch(vector<Cube> &ctr0, vector<Cube> &ctr1,
                                      vector<Cube> &ctrComp,
                                      vector<Cube> &ctrImage) {
#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP) or         \
    defined(USE_CLEARING_IMAGE)
    if (config.sparseComplex) {
        throw runtime_error(
            "sparse-complex dimension 1 is unaudited with USE_APPARENT_PAIRS, "
            "USE_APPARENT_PAIRS_COMP or USE_CLEARING_IMAGE");
    }
#endif
    // Sparse mode: append the cone columns once, BEFORE the parallel section
    // below (the three column lists are independent afterwards).
    if (cone != nullptr) {
        appendConeColumns(ctr0, cgc0);
        appendConeColumns(ctr1, cgc1);
        appendConeColumns(ctrComp, cgcComp);
    }

    auto processInput0 = [this, &ctr0]() {
#ifdef RUNTIME
        cout << endl << "input 0: ";
#endif
        computePairs(ctr0, 0);
#ifdef USE_CLEARING_DIM0
        enumerateEdges(ctr0, cgc0, pivotColumnIndexInput0);
#endif
    };

    auto processInput1 = [this, &ctr1]() {
#ifdef RUNTIME
        cout << endl << "input 1: ";
#endif

        computePairs(ctr1, 1);
#ifdef USE_CLEARING_DIM0
        enumerateEdges(ctr1, cgc1, pivotColumnIndexInput1);
#endif
    };

// The image worklist is a copy of the comparison columns. Hoisting that copy out
// of processComparison is what lets the image reductions start immediately
// instead of waiting for the comparison reduction (see the scheduling note
// below). It is value-identical ONLY because computePairsComp() does not mutate
// ctrComp: its single write site is the USE_CLEARING_IMAGE clearing pass. With
// that flag on, the copy must stay after the reduction and the barrier stays.
// The copy must in any case precede enumerateEdges(), which REPLACES ctrComp's
// 2-cells with dimension 0's edge worklist.
#ifdef DIM1_CTRIMAGE_HOISTED
    ctrImage = ctrComp;
#endif

    auto processComparison = [this, &ctrComp
#ifndef DIM1_CTRIMAGE_HOISTED
                              ,
                              &ctrImage
#endif
    ]() {
#ifdef RUNTIME
        cout << endl << "comparison: ";
#endif

        computePairsComp(ctrComp);
#ifdef USE_CLEARING_DIM0
#if not(defined(USE_APPARENT_PAIRS_COMP) or defined(DIM1_CTRIMAGE_HOISTED))
        ctrImage = ctrComp;
#endif
        enumerateEdges(ctrComp, cgcComp, pivotColumnIndexComp);
#endif
    };

    auto processImageInput0Comparison = [this,
#if not(defined(USE_APPARENT_PAIRS_COMP) or defined(USE_CLEARING_DIM0))
                                         &ctrComp,
#endif
                                         &ctrImage]() {
#ifdef RUNTIME
        cout << endl << "image 0: ";
#endif

#if defined(USE_APPARENT_PAIRS_COMP) or defined(USE_CLEARING_DIM0)
        computePairsImage(ctrImage, 0);
#else
        computePairsImage(ctrComp, 0);
#endif
    };

    auto processImageInput1Comparison = [this,
#if not(defined(USE_APPARENT_PAIRS_COMP) or defined(USE_CLEARING_DIM0))
                                         &ctrComp,
#endif
                                         &ctrImage]() {
#ifdef RUNTIME
        cout << endl << "image 1: ";
#endif

#if defined(USE_APPARENT_PAIRS_COMP) or defined(USE_CLEARING_DIM0)
        computePairsImage(ctrImage, 1);
#else
        computePairsImage(ctrComp, 1);
#endif
    };

#ifdef PARALLELIZE_INDEPENDENT_BARCODES_DIM1
    // SCHEDULING. The five reductions are independent -- they write disjoint
    // state (pairs/matchMap per side, pivotColumnIndex per pass) and are joined
    // only by computeMatching() below. The one real dependency was
    // `ctrImage = ctrComp`, and hoisting that copy above removes it, so all five
    // can now start at once.
    //
    // This matters because the reductions are heavily UNBALANCED (image 0
    // alone can be a large share of dim-1's work). With a
    // comparison-then-images barrier the critical path would be
    // comparison + image0; without it, it is max(...) = image0.
    //
    // USE_APPARENT_PAIRS_COMP still needs the barrier (it builds ctrImage inside
    // Dimension2/the comparison pass), and USE_ISPAIRED reads isPairedComp,
    // which the comparison pass writes. Both are compiled out by default.
    // NB: every future below is consumed with get(), not wait(). wait() blocks
    // but does NOT rethrow, and neither does std::future's destructor, so an
    // exception inside any of the five reductions would vanish silently and
    // the caller would receive a quietly truncated result.
    auto comparisonFuture = std::async(processComparison);
    auto input0Future = std::async(processInput0);
    auto input1Future = std::async(processInput1);
#ifndef DIM1_CTRIMAGE_HOISTED
    comparisonFuture.get();
#endif
    auto imageInput0ComparisonFuture = std::async(processImageInput0Comparison);
    auto imageInput1ComparisonFuture = std::async(processImageInput1Comparison);
    input0Future.get();
    input1Future.get();
    imageInput0ComparisonFuture.get();
    imageInput1ComparisonFuture.get();
#ifdef DIM1_CTRIMAGE_HOISTED
    comparisonFuture.get();
#endif
#else
    processInput0();
    processInput1();
    processComparison();
    processImageInput0Comparison();
    processImageInput1Comparison();
#endif

#ifdef RUNTIME
    cout << endl << "matching: ";
#endif
    computeMatching();
}

void Dimension1::computeInput0Pairs(vector<Cube> &ctr0) {
    if (cone != nullptr) {
        appendConeColumns(ctr0, cgc0);
    }
    computePairs(ctr0, 0);
#ifdef USE_CLEARING_DIM0
    enumerateEdges(ctr0, cgc0, pivotColumnIndexInput0);
#endif
}

void Dimension1::computePairs(vector<Cube> &ctr, uint8_t k) {
#ifdef USE_CACHE
    auto &cache = (k == 0) ? cacheInputPairs0 : cacheInputPairs1;
#endif
    computePairsUnified<ComputePairsMode::INPUT_PAIRS>(ctr, k,
#ifdef USE_CACHE
                                                       cache);
#endif
}

void Dimension1::computePairsComp(vector<Cube> &ctr) {
#ifdef USE_CACHE
    auto &cache = cacheCompPairs;
#endif
    computePairsUnified<ComputePairsMode::COMPARISON_PAIRS>(ctr, 0,
#ifdef USE_CACHE
                                                            cache);
#endif
}

void Dimension1::computePairsImage(vector<Cube> &ctr, uint8_t k) {
#ifdef USE_CACHE
    // The transient per-image cache is K-sized in sparse mode too.
    // Sparse mode: it must ALSO know about the cone, because the image
    // reductions cache cone 2-cell columns like any other (without it, every
    // write of a virtual key would throw).
    SparseOrDenseCubeMap<2, vector<Cube>> cache(cgc0.shape, kept, cone);
#endif
    computePairsUnified<ComputePairsMode::IMAGE_PAIRS>(ctr, k
#ifdef USE_CACHE
                                                       ,
                                                       cache
#endif
    );
}

void Dimension1::appendConeColumns(vector<Cube> &ctr,
                                   const CubicalGridComplex &cgc) const {
    if (cone == nullptr) {
        return;
    }
    // One column per (rim edge, pocket). With cycles built (pruning on)
    // only the NON-TREE slots are emitted, each carrying its explicit
    // fundamental cycle; the forest columns pair with cone edges at their own
    // birth and emit nothing, so they are not built at all.
    const bool cycles = cone->hasCycles();
    const size_t first = ctr.size();
    for (size_t i = 0; i < kept->keptEdges.size(); ++i) {
        const uint8_t count = cone->cone2CellCount(static_cast<int32_t>(i));
        if (count == 0) {
            continue;
        }
        const uint64_t key = kept->keptEdges[i];
        const index_t x = (key >> 44) & 0xfffff;
        const index_t y = (key >> 24) & 0xfffff;
        const index_t z = (key >> 4) & 0xfffff;
        const uint8_t type = key & 0xf;
        const int32_t base = cone->cone2CellSlot(static_cast<int32_t>(i), 0);
        for (uint8_t r = 0; r < count; ++r) {
            // Suppressed columns are simply not emitted; their SLOT still
            // exists, so every map sizing and every cone2CellSlot() stays
            // valid and the (birth, index) tie-break inside the virtual slice
            // is unperturbed.
            const int32_t slot = base + r;
            if (cycles ? !cone->emitCycleColumn(slot)
                       : !cone->emitCone2Cell(slot)) {
                continue;
            }
            // The V rule is the max over the column's faces; those are real
            // kept edges (with cycles) or the anchor edge plus two cone edges,
            // and a cone edge is born at CONE_BIRTH. BettiMatching has checked
            // that no kept cell exceeds CONE_BIRTH, so that max is CONE_BIRTH
            // unconditionally -- the same value in cgc0, cgc1 and cgcComp,
            // which is exactly what the shared virtual slice needs.
            ctr.push_back(Cube(CONE_BIRTH, x, y, z,
                               VIRTUAL_TYPE_BASE + 4 * type + r));
        }
    }
    // The cone block is APPENDED, not merged: at equal value, kept cells come
    // before virtual ones (config.h). Every real column is at most CONE_BIRTH,
    // so this is the (birth, kept-before-virtual, index) order, and "all of K
    // precedes the cone" holds in the total order even for real columns that
    // sit exactly at the background value. The block itself sorts by index.
    sort(ctr.begin() + first, ctr.end(), CubeComparator());
}

void Dimension1::computeMatching() {
#ifdef RUNTIME
    auto start = high_resolution_clock::now();
#endif

    uint64_t birthIndex0;
    uint64_t birthIndex1;
    for (Pair &pair : pairsComp) {
        auto find0 = matchMapIm0.find(pair.death.index);
        auto find1 = matchMapIm1.find(pair.death.index);
        if (find0.has_value() && find1.has_value()) {
            birthIndex0 = *find0;
            birthIndex1 = *find1;
            auto find0 = matchMap0.find(birthIndex0);
            auto find1 = matchMap1.find(birthIndex1);
            if (find0.has_value() && find1.has_value()) {
                // A censored comparison death (a cone column) joins two
                // CENSORED input bars, never a finite one, and a real
                // comparison death joins two FINITE ones. The censored
                // classes are H_1(K) -- the same space on all three sides,
                // since the three filtrations live on the same complex K and
                // all of K precedes the cone in every column order -- and the
                // image reductions reduce the same real columns as the
                // inputs' own reductions under the same row order, so the set
                // of rows the real block claims is identical: an edge killed
                // by a REAL column in an input's own reduction is claimed by
                // a real column in the image reduction too, and no cone
                // column can pivot on it. A "mixed" join therefore means the
                // three reductions disagree about which classes die at the
                // cone, which would be a bug in an earlier stage (e.g. an
                // unseeded comparison dual sweep), so it throws rather than
                // being routed.
                const bool censored0 =
                    cone != nullptr && isVirtualCell(find0->death.index);
                const bool censored1 =
                    cone != nullptr && isVirtualCell(find1->death.index);
                if (censored0 != censored1) {
                    throw runtime_error(
                        "pocket-cone dim-1 matching: a class is censored on "
                        "one input side and finite on the other; the three "
                        "reductions disagree on which classes die at the "
                        "cone");
                }
                if (censored0) {
                    auto c0 = censoredIndex0.find(find0->birth.index);
                    auto c1 = censoredIndex1.find(find1->birth.index);
                    if (c0 != censoredIndex0.end() &&
                        c1 != censoredIndex1.end()) {
                        essentialMatches.push_back({c0->second, c1->second});
                        isMatched0.emplace(find0->birth.index, true);
                        isMatched1.emplace(find1->birth.index, true);
                    }
                    continue;
                }
                matches.push_back(Match(*find0, *find1));
                isMatched0.emplace(find0->birth.index, true);
                isMatched1.emplace(find1->birth.index, true);
#ifdef COMPUTE_COMPARISON
                isMatchedWithIndexComp.emplace(pair.birth.index,
                                               matches.size() - 1);
#endif
            }
        }
    }

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms";
#endif
}

void Dimension1::enumerateEdges(
    vector<Cube> &edges, const CubicalGridComplex &cgc,
    SparseOrDenseCubeMap<1, size_t> &pivotColumnIndex) const {
#ifdef RUNTIME
    cout << "; enumeration ";
    auto start = high_resolution_clock::now();
#endif

    const bool sparse = config.sparseComplex && kept != nullptr;
    edges.clear();
    edges.reserve(sparse ? kept->keptEdges.size() : cgc.getNumberOfCubes(1));
    value_t birth;
#ifdef USE_CLEARING_DIM0
    Cube cube;
#endif
#ifdef USE_STABLE_SORT_OR_STABLE_PARTITION
    bool binaryInputs = true;
#endif
    if (sparse) {
        // Only kept edges can pass the birth filter (shared mask), so
        // iterate the kept list -- its canonical (x,y,z,type) order equals
        // the full-grid visit order restricted to kept edges.
        for (const uint64_t key : kept->keptEdges) {
            const index_t x = (key >> 44) & 0xfffff;
            const index_t y = (key >> 24) & 0xfffff;
            const index_t z = (key >> 4) & 0xfffff;
            const uint8_t type = key & 0xf;
            birth = cgc.getBirth(x, y, z, type, 1);
            if (birth < config.threshold) {
#ifdef USE_CLEARING_DIM0
                cube = Cube(birth, x, y, z, type);
                auto find = pivotColumnIndex.find(cube.index);
                if (!find.has_value()) {
                    edges.push_back(cube);
#ifdef USE_STABLE_SORT_OR_STABLE_PARTITION
                    if (binaryInputs && birth != 0 && birth != 1)
                        binaryInputs = false;
#endif
                }
#else
                edges.push_back(Cube(birth, x, y, z, type));
#ifdef USE_STABLE_SORT_OR_STABLE_PARTITION
                if (binaryInputs && birth != 0 && birth != 1)
                    binaryInputs = false;
#endif
#endif
            }
        }
    } else {
        for (index_t x = 0; x < cgc.shape[0]; ++x) {
            for (index_t y = 0; y < cgc.shape[1]; ++y) {
                for (index_t z = 0; z < cgc.shape[2]; ++z) {
                    for (uint8_t type = 0; type < 3; ++type) {
                        birth = cgc.getBirth(x, y, z, type, 1);
                        if (birth < config.threshold) {
#ifdef USE_CLEARING_DIM0
                            cube = Cube(birth, x, y, z, type);
                            auto find = pivotColumnIndex.find(cube.index);
                            if (!find.has_value()) {
                                edges.push_back(cube);
#ifdef USE_STABLE_SORT_OR_STABLE_PARTITION
                                if (binaryInputs && birth != 0 && birth != 1)
                                    binaryInputs = false;
#endif
                            }
#else
                            edges.push_back(Cube(birth, x, y, z, type));
#ifdef USE_STABLE_SORT_OR_STABLE_PARTITION
                            if (binaryInputs && birth != 0 && birth != 1)
                                binaryInputs = false;
#endif
#endif
                        }
                    }
                }
            }
        }
    }
#ifdef USE_STABLE_SORT_OR_STABLE_PARTITION
    if (binaryInputs) {
        std::stable_partition(edges.begin(), edges.end(),
                              [](Cube &cube) { return cube.birth == 0; });
    } else {
        std::stable_sort(edges.begin(), edges.end(),
                         [](const Cube &cube1, const Cube &cube2) {
                             return cube1.birth < cube2.birth;
                         });
    }
#else
    std::sort(edges.begin(), edges.end(), CubeComparator());
#endif

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms";
#endif
}

void Dimension1::enumerateColumnsToReduce(vector<Cube> &ctr,
                                          const CubicalGridComplex &cgc) const {
    ctr.reserve(cgc.getNumberOfCubes(2));
    value_t birth;
    for (index_t x = 0; x < cgc.shape[0]; ++x) {
        for (index_t y = 0; y < cgc.shape[1]; ++y) {
            for (index_t z = 0; z < cgc.shape[2]; ++z) {
                for (uint8_t type = 0; type < 3; ++type) {
                    birth = cgc.getBirth(x, y, z, type, 2);
                    if (birth < config.threshold) {
                        ctr.push_back(Cube(birth, x, y, z, type));
                    }
                }
            }
        }
    }
    sort(ctr.begin(), ctr.end(), CubeComparator());
}

Cube Dimension1::popPivot(CubeQueue &column) const {
    if (column.empty()) {
        return Cube();
    } else {
        Cube pivot = column.top();
        column.pop();
        while (!column.empty() && column.top() == pivot) {
            column.pop();
            if (column.empty()) {
                return Cube();
            } else {
                pivot = column.top();
                column.pop();
            }
        }
        return pivot;
    }
}

Cube Dimension1::getPivot(CubeQueue &column) const {
    Cube result = popPivot(column);
    if (result.index != NONE_INDEX) {
        column.push(result);
    }
    return result;
}

#ifdef USE_REDUCTION_MATRIX
void Dimension1::useReductionMatrix(const Cube &column,
                                    CubeQueue &workingBoundary,
                                    BoundaryEnumerator &enumerator
#ifdef USE_CACHE
                                    ,
                                    SparseOrDenseCubeMap<2, vector<Cube>> &cache
#endif
) const {
    auto reductionColumn = reductionMatrix.find(column.index);
    if (reductionColumn.has_value()) {
        for (Cube &row : *reductionColumn) {
#ifdef USE_CACHE
            if (!columnIsCached(row, workingBoundary, cache)) {
#endif
                enumerator.setBoundaryEnumerator(row);
                while (enumerator.hasNextFace()) {
                    workingBoundary.push(enumerator.nextFace);
                }
                useReductionMatrix(row, workingBoundary, enumerator
#ifdef USE_CACHE
                                   ,
                                   cache
#endif
                );
#ifdef USE_CACHE
            }
#endif
        }
    }
}
#endif

#ifdef USE_CACHE
bool Dimension1::columnIsCached(
    const Cube &column, CubeQueue &workingBoundary,
    SparseOrDenseCubeMap<2, vector<Cube>> &cache) const {
    auto &cachedBoundary = cache.find(column.index);
    if (cachedBoundary.has_value()) {
        for (auto &face : *cachedBoundary) {
            workingBoundary.push(face);
        }
        return true;
    } else {
        return false;
    }
}

void Dimension1::addCache(const Cube &column, CubeQueue &workingBoundary,
                          queue<uint64_t> &cachedColumnIdx,
                          SparseOrDenseCubeMap<2, vector<Cube>> &cache) {
    std::vector<Cube> cleanWb;
    while (!workingBoundary.empty()) {
        Cube c = workingBoundary.top();
        workingBoundary.pop();
        if (!workingBoundary.empty() && c == workingBoundary.top()) {
            workingBoundary.pop();
        } else {
            cleanWb.emplace_back(c);
        }
    }

    cache[column.index] = std::move(cleanWb);
    cachedColumnIdx.push(column.index);
    if (cachedColumnIdx.size() > config.cacheSize) {
        cache[cachedColumnIdx.front()] = {};
        cachedColumnIdx.pop();
    }
}
#endif

#ifdef USE_EMERGENT_PAIRS
template <Dimension1::ComputePairsMode computePairsMode>
bool Dimension1::isEmergentPair(
    const Cube &column, Cube &pivot, size_t &j, vector<Cube> &faces,
    bool &checkEmergentPair, const CubicalGridComplex &cgc,
    BoundaryEnumerator &enumerator, BoundaryEnumerator &enumeratorAP,
    CoboundaryEnumerator &coEnumeratorAP,
    SparseOrDenseCubeMap<1, size_t> &pivotColumnIndex) const {
    // The image reduction is the ONLY one that re-derives a column's birth from
    // the grid (the other three use column.birth directly). A virtual cone
    // 2-cell has no grid birth: it is born at CONE_BIRTH in every complex
    // (getBirth throws on a virtual cell).
    auto birth = column.birth;
    if (computePairsMode == IMAGE_PAIRS) {
        if (cone != nullptr && isVirtualCell(column.index)) {
            birth = CONE_BIRTH;      // see appendConeColumns
        } else {
            birth = cgc.getBirth(column.x(), column.y(), column.z(),
                                 column.type(), 2);
        }
    }

    const bool useApparentPairs =
#ifdef USE_APPARENT_PAIRS
        true;
#else
        false;
#endif
    const bool useApparentPairsComp =
#ifdef USE_APPARENT_PAIRS_COMP
        true;
#else
        false;
#endif

    faces.clear();
    enumerator.setBoundaryEnumerator(column);
    while (enumerator.hasPreviousFace()) {
        if (checkEmergentPair && enumerator.nextFace.birth == birth) {
            auto nextColumnIndex =
                pivotColumnIndex.find(enumerator.nextFace.index);
            if ((useApparentPairs && computePairsMode == INPUT_PAIRS) ||
                (useApparentPairsComp &&
                 computePairsMode == COMPARISON_PAIRS) ||
                computePairsMode == IMAGE_PAIRS) {
                if (nextColumnIndex.has_value()) {
                    checkEmergentPair = false;
                    j = *nextColumnIndex;
                }
#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP)
                else if ((computePairsMode == INPUT_PAIRS ||
                          computePairsMode == COMPARISON_PAIRS) &&
                         pivotOfColumnIsApparentPair(
                             enumerator.nextFace, column, faces, enumeratorAP,
                             coEnumeratorAP)) {
                    checkEmergentPair = false;
                }
#endif
                else {
                    pivot = enumerator.nextFace;
                    return true;
                }
            } else if (computePairsMode == INPUT_PAIRS ||
                       computePairsMode == COMPARISON_PAIRS) {
                if (nextColumnIndex.has_value()) {
                    checkEmergentPair = false;
                    j = *nextColumnIndex;
                } else {
                    pivot = enumerator.nextFace;
                    return true;
                }
            }
        }
        faces.push_back(enumerator.nextFace);
    }
    return false;
}
#endif

#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP)
bool Dimension1::pivotIsApparentPair(const Cube &pivot, vector<Cube> &faces,
                                     BoundaryEnumerator &enumerator,
                                     CoboundaryEnumerator &coEnumerator) const {
    bool foundApparentPair = false;
    coEnumerator.setCoboundaryEnumerator(pivot);
    while (coEnumerator.hasNextCoface()) {
        if (coEnumerator.nextCoface.birth == pivot.birth) {
            vector<Cube> facesCopy = faces;
            enumerator.setBoundaryEnumerator(coEnumerator.nextCoface);
            while (enumerator.hasPreviousFace()) {
                if (enumerator.nextFace == pivot) {
                    foundApparentPair = true;
                } else if (!foundApparentPair &&
                           enumerator.nextFace.birth ==
                               coEnumerator.nextCoface.birth) {
                    return false;
                }
                facesCopy.push_back(enumerator.nextFace);
            }
            if (foundApparentPair) {
                faces = facesCopy;
            }
            break;
        }
    }
    return foundApparentPair;
}

bool Dimension1::pivotOfColumnIsApparentPair(
    const Cube &pivot, const Cube &column, vector<Cube> &faces,
    BoundaryEnumerator &enumerator, CoboundaryEnumerator &coEnumerator) const {
    bool foundApparentPair = false;
    coEnumerator.setCoboundaryEnumerator(pivot);
    while (coEnumerator.hasNextCoface()) {
        if (coEnumerator.nextCoface == column) {
            break;
        }
        if (coEnumerator.nextCoface.birth == pivot.birth) {
            vector<Cube> facesCopy = faces;
            enumerator.setBoundaryEnumerator(coEnumerator.nextCoface);
            while (enumerator.hasPreviousFace()) {
                if (enumerator.nextFace == pivot) {
                    foundApparentPair = true;
                } else if (!foundApparentPair &&
                           enumerator.nextFace.birth ==
                               coEnumerator.nextCoface.birth) {
                    return false;
                }
                facesCopy.push_back(enumerator.nextFace);
            }
            if (foundApparentPair) {
                faces = facesCopy;
            }
            break;
        }
    }
    return foundApparentPair;
}
#endif

template <Dimension1::ComputePairsMode computePairsMode>
void Dimension1::computePairsUnified(vector<Cube> &ctr, uint8_t k
#ifdef USE_CACHE
                                     ,
                                     SparseOrDenseCubeMap<2, vector<Cube>> &cache
#endif
) {
#ifdef RUNTIME
    cout << "barcode ";
    auto start = high_resolution_clock::now();
#endif
    const CubicalGridComplex &cgc = (computePairsMode == COMPARISON_PAIRS)
                                        ? cgcComp
                                        : ((k == 0) ? cgc0 : cgc1);
    vector<Pair> &pairs = (computePairsMode == COMPARISON_PAIRS)
                              ? pairsComp
                              : ((k == 0) ? pairs0 : pairs1);
    SparseOrDenseCubeMap<1, Pair> &matchMap =
        (computePairsMode == INPUT_PAIRS)
            ? ((k == 0) ? matchMap0 : matchMap1)
            : matchMap0;
    SparseOrDenseCubeMap<2, uint64_t> &matchMapIm =
        (computePairsMode == IMAGE_PAIRS)
            ? ((k == 0) ? matchMapIm0 : matchMapIm1)
            : matchMapIm0;
    SparseOrDenseCubeMap<1, size_t> &pivotColumnIndex =
        (computePairsMode == INPUT_PAIRS)
            ? ((k == 0) ? pivotColumnIndexInput0 : pivotColumnIndexInput1)
        : (computePairsMode == COMPARISON_PAIRS)
            ? pivotColumnIndexComp
            : ((k == 0) ? pivotColumnIndexImage0 : pivotColumnIndexImage1);

#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP)
    const bool useApparentPairs =
#ifdef USE_APPARENT_PAIRS
        true;
#else
        false;
#endif
#endif
#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP) or         \
    defined(USE_CLEARING_IMAGE)
    const bool useApparentPairsComp =
#ifdef USE_APPARENT_PAIRS_COMP
        true;
#else
        false;
#endif
#endif

    size_t ctrSize = ctr.size();
    BoundaryEnumerator enumerator(cgc, kept, cone, CONE_BIRTH);
    Cube pivot;
    size_t j;
#ifdef USE_REDUCTION_MATRIX
    reductionMatrix.clear();
    vector<Cube> reductionColumn;
#ifdef RUNTIME
    size_t numReductionColumns = 0;
#endif
#endif
#ifdef USE_CACHE
    queue<uint64_t> cachedColumnIdx;
    size_t numRecurse;
#ifdef RUNTIME
    size_t numCached = 0;
#endif
#endif
#ifdef USE_EMERGENT_PAIRS
    bool checkEmergentPair;
#ifdef RUNTIME
    size_t numEmergentPairs = 0;
#endif
#endif
#ifdef RUNTIME
    // Column-outcome census: does a column end on a PIVOT (a dim-1 pair) or at
    // ZERO (it bounds), and how many merge steps did it take to find out?
    size_t nZero = 0, nPivot = 0, hopsZero = 0, hopsPivot = 0;
    size_t histZero[8] = {0}, histPivot[8] = {0};
    // Emergent-pair BLOCKING census: an emergent pair fails when the
    // column's youngest same-birth face is already a pivot. Record, per
    // (column is virtual?, blocking column is virtual?), how often that
    // happens and how far back the blocker sits in ctr.
    size_t nBlock[2][2] = {{0, 0}, {0, 0}};
    double distBlock[2][2] = {{0, 0}, {0, 0}};
    size_t histBlock[2][2][8] = {{{0}}};
    size_t nNoFace[2] = {0, 0};       // no same-birth face at all
    size_t nEmergentBy[2] = {0, 0};   // emergent pairs, by column virtuality
    size_t nColBy[2] = {0, 0};        // columns entering, by virtuality
    // Pivot outcome by (column virtual?, pivot cell virtual?), with hops.
    size_t nPivotBy[2][2] = {{0, 0}, {0, 0}};
    size_t hopsPivotBy[2][2] = {{0, 0}, {0, 0}};
    size_t nZeroBy[2] = {0, 0};
    size_t hopsZeroBy[2] = {0, 0};
    size_t nNonzeroPersBy[2] = {0, 0};   // pivot.birth != column.birth
    size_t nEmergentNonzeroBy[2] = {0, 0};
    auto bucket = [](size_t r) -> int {
        if (r < 3) { return static_cast<int>(r); }
        if (r < 5) { return 3; }
        if (r < 9) { return 4; }
        if (r < 17) { return 5; }
        if (r < 65) { return 6; }
        return 7;
    };
#endif
    vector<Cube> faces;
    BoundaryEnumerator enumeratorAP(cgc);
    CoboundaryEnumerator coEnumeratorAP(cgc);
    if (computePairsMode == COMPARISON_PAIRS) {
#if defined(USE_CLEARING_IMAGE) and not defined(USE_APPARENT_PAIRS_COMP)
        shouldClear = false;
#endif
    }
    if (computePairsMode == IMAGE_PAIRS) {
#ifdef USE_CLEARING_IMAGE
        shouldClear = false;
#endif
    }

    for (size_t i = 0; i < ctrSize; ++i) {
        CubeQueue workingBoundary;
        j = i;
#ifdef RUNTIME
        const int colV =
            (cone != nullptr && isVirtualCell(ctr[i].index)) ? 1 : 0;
        ++nColBy[colV];
#endif
#ifdef USE_CACHE
        numRecurse = 0;
#endif
#ifdef USE_EMERGENT_PAIRS
        checkEmergentPair = true;
#endif
        while (true) {
            if (j == i) {
#ifdef USE_EMERGENT_PAIRS
                if (isEmergentPair<computePairsMode>(
                        ctr[i], pivot, j, faces, checkEmergentPair, cgc,
                        enumerator, enumeratorAP, coEnumeratorAP,
                        pivotColumnIndex)) {
                    pivotColumnIndex.emplace(pivot.index, i);

                    if (computePairsMode == IMAGE_PAIRS
#ifdef USE_ISPAIRED
                        && isPairedComp[ctr[i].index]
#endif
                    ) {
                        matchMapIm.emplace(ctr[i].index, pivot.index);
                    }

#ifdef RUNTIME
                    ++numEmergentPairs;
                    ++nEmergentBy[colV];
                    if (pivot.birth != ctr[i].birth) {
                        ++nEmergentNonzeroBy[colV];
                    }
#endif
                    break;
                } else {
#ifdef RUNTIME
                    if (j != i) {
                        const int blkV =
                            (cone != nullptr && isVirtualCell(ctr[j].index))
                                ? 1
                                : 0;
                        ++nBlock[colV][blkV];
                        distBlock[colV][blkV] +=
                            static_cast<double>(i - j);
                        ++histBlock[colV][blkV][bucket(i - j)];
                    } else {
                        ++nNoFace[colV];
                    }
#endif
                    for (auto face = faces.rbegin(), last = faces.rend();
                         face != last; ++face) {
                        workingBoundary.push(*face);
                    }
#ifdef USE_CACHE
                    ++numRecurse;
#endif
                    if (j != i) {
                        continue;
                    }
#ifdef USE_REDUCTION_MATRIX
                    else if (computePairsMode == INPUT_PAIRS ||
                             computePairsMode == COMPARISON_PAIRS) {
                        reductionColumn.push_back(coEnumeratorAP.nextCoface);
                    }
#endif
                }
#else
                enumerator.setBoundaryEnumerator(ctr[i]);
                while (enumerator.hasNextFace()) {
                    workingBoundary.push(enumerator.nextFace);
                }
#endif
            } else {
#ifdef USE_REDUCTION_MATRIX
                reductionColumn.push_back(ctr[j]);
#endif
#ifdef USE_CACHE
                if (!columnIsCached(ctr[j], workingBoundary, cache)) {
#endif
                    enumerator.setBoundaryEnumerator(ctr[j]);
                    while (enumerator.hasNextFace()) {
                        workingBoundary.push(enumerator.nextFace);
                    }
#ifdef USE_REDUCTION_MATRIX
                    useReductionMatrix(ctr[j], workingBoundary, enumerator
#ifdef USE_CACHE
                                       ,
                                       cache
#endif
                    );
#endif
#ifdef USE_CACHE
                }
#endif
            }
            pivot = getPivot(workingBoundary);

#if defined(USE_APPARENT_PAIRS) or defined(USE_APPARENT_PAIRS_COMP)
            if ((useApparentPairs && computePairsMode == INPUT_PAIRS) ||
                (useApparentPairsComp &&
                 computePairsMode == COMPARISON_PAIRS)) {
                while (true) {
                    faces.clear();
                    if (pivotIsApparentPair(pivot, faces, enumeratorAP,
                                            coEnumeratorAP)) {
                        for (auto face = faces.rbegin(), last = faces.rend();
                             face != last; ++face) {
                            workingBoundary.push(*face);
                        }
#ifdef USE_REDUCTION_MATRIX
                        reductionColumn.push_back(coEnumeratorAP.nextCoface);
#endif
#ifdef USE_CACHE
                        ++numRecurse;
#endif
                        pivot = getPivot(workingBoundary);
                    } else {
                        break;
                    }
                }
            }
#endif
            if (pivot.index != NONE_INDEX) {
                auto cachedIndex = pivotColumnIndex.find(pivot.index);
                if (cachedIndex.has_value()) {
                    j = *cachedIndex;
#ifdef USE_CACHE
                    ++numRecurse;
#endif
                    continue;
                } else {
#ifdef RUNTIME
                    ++nPivot;
                    hopsPivot += numRecurse;
                    ++histPivot[bucket(numRecurse)];
                    {
                        const int pvV =
                            (cone != nullptr && isVirtualCell(pivot.index))
                                ? 1
                                : 0;
                        ++nPivotBy[colV][pvV];
                        hopsPivotBy[colV][pvV] += numRecurse;
                        if (pivot.birth != ctr[i].birth) {
                            ++nNonzeroPersBy[colV];
                        }
                    }
#endif
                    pivotColumnIndex.emplace(pivot.index, i);
                    if (pivot.birth != ctr[i].birth) {
                        // Every bar the reduction emits reaches the matching;
                        // a kept edge at or above tau is a real edge of the
                        // filtration at a voxel the other input puts below
                        // threshold.
                        const bool censored =
                            cone != nullptr && isVirtualCell(ctr[i].index);
                        if (computePairsMode == INPUT_PAIRS) {
                            matchMap.emplace(pivot.index, Pair(pivot, ctr[i]));
                            if (censored) {
                                // The death column is a cone 2-cell at
                                // CONE_BIRTH with no voxel. Surface the BIRTH
                                // edge through the essentials head, whose loss
                                // semantics is "no death term, gradient on the
                                // birth voxel only".
                                vector<Cube> &ess =
                                    (k == 0) ? essentials0 : essentials1;
                                unordered_map<uint64_t, size_t> &cix =
                                    (k == 0) ? censoredIndex0 : censoredIndex1;
                                ess.push_back(pivot);
                                cix.emplace(pivot.index, ess.size() - 1);
                            } else {
                                pairs.push_back(Pair(pivot, ctr[i]));
                            }
                        } else if (computePairsMode == COMPARISON_PAIRS) {
                            // A censored comparison pair MUST still be pushed:
                            // computeMatching() iterates pairsComp to drive the
                            // death-cube join, so skipping it silently loses
                            // every censored match. Its virtual death cell is
                            // never dereferenced -- computeVoxels only reads a
                            // comparison pair's death for entries recorded in
                            // isMatchedWithIndexComp, which censored matches
                            // deliberately do not set.
                            pairs.push_back(Pair(pivot, ctr[i]));
#ifdef USE_ISPAIRED
                            isPairedComp.emplace(ctr[i].index, true);
#endif
                        }
                    }
                    if (computePairsMode == IMAGE_PAIRS
#ifdef USE_ISPAIRED
                        && isPairedComp[ctr[i].index]
#endif
                    ) {
                        matchMapIm.emplace(ctr[i].index, pivot.index);
                    }
#ifdef USE_CACHE
                    if (numRecurse >= config.minRecursionToCache) {
                        addCache(ctr[i], workingBoundary, cachedColumnIdx,
                                 cache);
#ifdef RUNTIME
                        ++numCached;
#endif
                        break;
                    }
#endif
#ifdef USE_REDUCTION_MATRIX
                    if (reductionColumn.size() > 0) {
                        reductionMatrix.emplace(ctr[i].index, reductionColumn);
                        reductionColumn.clear();
#ifdef RUNTIME
                        ++numReductionColumns;
#endif
                    }
#endif
                    break;
                }
            } else {
#ifdef RUNTIME
                ++nZero;
                hopsZero += numRecurse;
                ++histZero[bucket(numRecurse)];
                ++nZeroBy[colV];
                hopsZeroBy[colV] += numRecurse;
#endif
#if defined(USE_CLEARING_IMAGE)
                if ((computePairsMode == COMPARISON_PAIRS &&
                     !useApparentPairsComp) ||
                    computePairsMode == IMAGE_PAIRS) {
                    ctr[i].index = NONE_INDEX;
                    shouldClear = true;
                }
#endif
                break;
            }
        }
    }

#if defined(USE_CLEARING_IMAGE)
    if ((computePairsMode == COMPARISON_PAIRS && !useApparentPairsComp) ||
        computePairsMode == IMAGE_PAIRS) {
        if (shouldClear) {
            auto newEnd =
                remove_if(ctr.begin(), ctr.end(), [](const Cube &cube) {
                    return cube.index == NONE_INDEX;
                });
            ctr.erase(newEnd, ctr.end());
        }
    }
#endif

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms";
#ifdef USE_REDUCTION_MATRIX
    cout << ", " << numReductionColumns << " reduction columns";
#endif
#ifdef USE_CACHE
    cout << ", " << numCached << " cached columns";
#endif
#ifdef USE_EMERGENT_PAIRS
    cout << ", " << numEmergentPairs << " emergent pairs";
    {
        auto dump = [](const char *tag, size_t n, size_t hops,
                       const size_t *h) {
            cout << "\n      " << tag << " n=" << n << " hops=" << hops
                 << " mean=" << (n ? double(hops) / double(n) : 0.0)
                 << " hist[0,1,2,3-4,5-8,9-16,17-64,65+]=";
            for (int b = 0; b < 8; ++b) {
                cout << h[b] << (b < 7 ? "/" : "");
            }
        };
        dump("ENDS-ON-PIVOT ", nPivot, hopsPivot, histPivot);
        dump("ENDS-AT-ZERO  ", nZero, hopsZero, histZero);
        static const char *cn[2] = {"REAL-col", "VIRT-col"};
        static const char *bn[2] = {"blocked-by-REAL", "blocked-by-VIRT"};
        for (int c = 0; c < 2; ++c) {
            cout << "\n      " << cn[c] << " entering=" << nColBy[c]
                 << " emergent=" << nEmergentBy[c]
                 << " noSameBirthFace=" << nNoFace[c];
            for (int b = 0; b < 2; ++b) {
                cout << "\n        " << cn[c] << " " << bn[b]
                     << " n=" << nBlock[c][b] << " meanDist="
                     << (nBlock[c][b]
                             ? distBlock[c][b] / double(nBlock[c][b])
                             : 0.0)
                     << " dist_hist=";
                for (int q = 0; q < 8; ++q) {
                    cout << histBlock[c][b][q] << (q < 7 ? "/" : "");
                }
            }
            cout << "\n        " << cn[c]
                 << " pivot-on-REAL n=" << nPivotBy[c][0]
                 << " hops=" << hopsPivotBy[c][0]
                 << " | pivot-on-VIRT n=" << nPivotBy[c][1]
                 << " hops=" << hopsPivotBy[c][1]
                 << " | ends-at-ZERO n=" << nZeroBy[c]
                 << " hops=" << hopsZeroBy[c]
                 << " || NONZERO-persistence searched=" << nNonzeroPersBy[c]
                 << " emergent=" << nEmergentNonzeroBy[c];
        }
    }
#endif
#endif
}

RepresentativeCycle cubeCycleToVoxelCycle(vector<Cube> &cubeCycle) {
    // This function deduplicates voxels in obvious consecutive cases, but it is
    // not necessarily guaranteed that its output does not contain duplicate
    // voxels.
    if (cubeCycle.empty()) {
        return {};
    }
    auto &cube = cubeCycle[0];
    RepresentativeCycle voxelCycle{{cube.x(), cube.y(), cube.z()},
                                   {cube.x() + (cube.type() == 0),
                                    cube.y() + (cube.type() == 1),
                                    cube.z() + (cube.type() == 2)}};

    for (auto &cube : cubeCycle) {
        tuple<index_t, index_t, index_t> endpoint0 = {cube.x(), cube.y(),
                                                      cube.z()};
        auto type = cube.type();
        tuple<index_t, index_t, index_t> endpoint1 = {cube.x() + (type == 0),
                                                      cube.y() + (type == 1),
                                                      cube.z() + (type == 2)};
        auto lastAdded1 = voxelCycle[voxelCycle.size() - 1];
        auto lastAdded2 = voxelCycle[voxelCycle.size() - 2];
        if (lastAdded1 != endpoint0 && lastAdded2 != endpoint0) {
            voxelCycle.emplace_back(std::move(endpoint0));
        }
        if (lastAdded1 != endpoint1 && lastAdded2 != endpoint1) {
            voxelCycle.emplace_back(std::move(endpoint1));
        }
    }

    return voxelCycle;
}

vector<dim3::RepresentativeCycle> Dimension1::computeRepresentativeCycles(
    const int input,
    const std::vector<std::reference_wrapper<Pair>> &requestedPairs) {
#ifndef USE_CACHE
    throw runtime_error("computeRepresentativeCycles() can only be used when "
                        "compiled with USE_CACHE defined");
#endif
    auto &cache = (input == 0)   ? cacheInputPairs0
                  : (input == 1) ? cacheInputPairs1
                                 : cacheCompPairs;
    const CubicalGridComplex &cgc = (input == 0)   ? cgc0
                                    : (input == 1) ? cgc1
                                                   : cgcComp;

    vector<RepresentativeCycle> representativeCycles;

    // Compute representative cycles for requested pairs, using the cached
    // boundaries
    for (auto &requestedPair : requestedPairs) {
        auto &cachedBoundary = cache[requestedPair.get().death.index];
        if (!cachedBoundary.has_value()) {
            throw runtime_error(
                "A boundary that is needed to get all cycles was deleted from "
                "cache! Consider increasing the cache size limit.");
        }
        auto representativeCycle = cubeCycleToVoxelCycle(*cachedBoundary);
        representativeCycle.push_back(
            cgc.getParentVoxel(requestedPair.get().death, 2));
        representativeCycles.emplace_back(std::move(representativeCycle));
    }

    return representativeCycles;
}
