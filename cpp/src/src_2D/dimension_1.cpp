#include "dimension_1.h"
#include "data_structures.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <set>

using namespace dim2;
using namespace std;
using namespace std::chrono;

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
                       const KeptCells *_kept, const PocketNodes *_pockets)
    : cgc0(_cgc0), cgc1(_cgc1), cgcComp(_cgcComp), config(_config),
      pairs0(_pairs0), pairs1(_pairs1), pairsComp(_pairsComp),
      matches(_matches), isMatched0(_isMatched0), isMatched1(_isMatched1),
      isMatchedWithIndexComp(_isMatchedWithIndexComp),
      essentials0(_essentials0), essentials1(_essentials1),
      essentialMatches(_essentialMatches), kept(_kept), pockets(_pockets),
      uf0(UnionFindDual(cgc0, _kept, _pockets, CONE_BIRTH)),
      uf1(UnionFindDual(cgc1, _kept, _pockets, CONE_BIRTH)),
      ufComp(UnionFindDual(cgcComp, _kept, _pockets, CONE_BIRTH)) {
    // Pockets are born at CONE_BIRTH in ALL THREE complexes: the masked region
    // is the last thing to enter every one of them -- at or above every kept
    // cell (checked in BettiMatching), and elder to every kept top cell at a
    // tie by link()'s original-index rule. In 2D dim 1 IS the top
    // dimension, so this dual union-find is the whole of the sparse
    // construction -- there are no cone 2-cells to add.
    uf0.seedPockets();
    uf1.seedPockets();
    ufComp.seedPockets();
}

void Dimension1::computePairsAndMatch(vector<Cube> &ctr0, vector<Cube> &ctr1,
                                      vector<Cube> &ctrComp) {
#ifdef RUNTIME
    cout << endl << "input & image 0: ";
#endif
    enumerateDualEdges(ctr0, cgc0);
    computeInputAndImagePairs(ctr0, 0);

#ifdef RUNTIME
    cout << endl << "input & image 1: ";
#endif
    enumerateDualEdges(ctr1, cgc1);
    ufComp.reset();
    ufComp.seedPockets();
    computeInputAndImagePairs(ctr1, 1);

#ifdef RUNTIME
    cout << endl << "comparison & matching: ";
#endif
    enumerateDualEdges(ctrComp, cgcComp);
    ufComp.reset();
    ufComp.seedPockets();
    computeCompPairsAndMatch(ctrComp);
}

void Dimension1::computeInput0Pairs(vector<Cube> &ctr0) {
    enumerateDualEdges(ctr0, cgc0);
    computeInputAndImagePairs(ctr0, 0);
}

void Dimension1::enumerateDualEdges(vector<Cube> &dualEdges,
                                    const CubicalGridComplex &cgc) const {
#ifdef RUNTIME
    cout << "enumeration ";
    auto start = high_resolution_clock::now();
#endif

    const bool sparse = config.sparseComplex && kept != nullptr;
    dualEdges.reserve(sparse ? kept->keptEdges.size()
                             : cgc.getNumberOfCubes(1));
    value_t birth;
    if (sparse) {
        // Only kept edges can pass the birth filter (shared mask), so
        // iterate the kept list -- its canonical (x,y,type) order equals the
        // full-grid visit order restricted to kept cells (and the sort's
        // (birth, index) order is total, so the result is identical).
        for (const uint64_t key : kept->keptEdges) {
            const index_t x = (key >> 34) & 0xfffff;
            const index_t y = (key >> 4) & 0xfffff;
            const uint8_t type = key & 0xf;
            birth = cgc.getBirth(x, y, type, 1);
            if (birth < config.threshold) {
                dualEdges.push_back(Cube(birth, x, y, type));
            }
        }
    } else {
        for (index_t x = 0; x < cgc.shape[0]; ++x) {
            for (index_t y = 0; y < cgc.shape[1]; ++y) {
                for (uint8_t type = 0; type < 2; ++type) {
                    birth = cgc.getBirth(x, y, type, 1);
                    if (birth < config.threshold) {
                        dualEdges.push_back(Cube(birth, x, y, type));
                    }
                }
            }
        }
    }

    sort(dualEdges.begin(), dualEdges.end(), CubeComparator());

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms, ";
#endif
}

void Dimension1::computeInputAndImagePairs(vector<Cube> &dualEdges,
                                           const uint8_t &k) {
#ifdef RUNTIME
    cout << "barcodes ";
    auto start = high_resolution_clock::now();
#endif

    UnionFindDual &uf = (k == 0) ? uf0 : uf1;
    vector<Pair> &pairs = (k == 0) ? pairs0 : pairs1;
    unordered_map<index_t, Pair> &matchMap = (k == 0) ? matchMap0 : matchMap1;
    vector<Cube> &essentials = (k == 0) ? essentials0 : essentials1;
    unordered_map<index_t, size_t> &censoredMatchMap =
        (k == 0) ? censoredMatchMap0 : censoredMatchMap1;
    vector<index_t> boundaryIndices;
    index_t parentIdx0;
    index_t parentIdx1;
    index_t birthIdx;
    index_t birthIdxComp;
    value_t birth;
    dim2::Coordinate birthCoordinates;
    for (auto edge = dualEdges.rbegin(), last = dualEdges.rend(); edge != last;
         ++edge) {
        // Sparse: masked top cells resolve to their pocket's virtual node.
        boundaryIndices = uf.getBoundaryIndices(*edge);
        parentIdx0 = uf.find(boundaryIndices[0]);
        parentIdx1 = uf.find(boundaryIndices[1]);
        if (parentIdx0 != parentIdx1) {
            birthIdx = uf.link(parentIdx0, parentIdx1);
            birth = uf.getBirth(birthIdx);
            parentIdx0 = ufComp.find(boundaryIndices[0]);
            parentIdx1 = ufComp.find(boundaryIndices[1]);
            birthIdxComp = ufComp.link(parentIdx0, parentIdx1);
            if (edge->birth != birth) {
                if (uf.isPocketNode(birthIdx)) {
                    // CENSORED: the dying root is a virtual pocket, so the
                    // death is CONE_BIRTH and there is no death voxel. A
                    // pocket is elder to every kept top cell (at or above it,
                    // and the original-index tie-break lets the cube die), so
                    // this fires only for pocket-vs-pocket and
                    // pocket-vs-sentinel merges -- exactly the genuine cavity
                    // deaths. Record the BIRTH cell; the loss's essential head
                    // already means "death censored, gradient on the birth
                    // voxel only", for every birth value (see
                    // src_3D/dimension_2.cpp).
                    essentials.push_back(*edge);
                    censoredMatchMap.emplace(birthIdxComp,
                                             essentials.size() - 1);
                } else {
                    birthCoordinates = uf.getCoordinates(birthIdx);
                    pairs.push_back(
                        Pair(*edge, Cube(birth, std::get<0>(birthCoordinates),
                                         std::get<1>(birthCoordinates), 0)));
                    matchMap.emplace(birthIdxComp, pairs.back());
                }
            }
#ifdef USE_CLEARING_DIM0
            edge->index = NONE_INDEX;
#endif
        }
    }

#ifdef USE_CLEARING_DIM0
    auto new_end =
        remove_if(dualEdges.begin(), dualEdges.end(),
                  [](const Cube &cube) { return cube.index == NONE_INDEX; });
    dualEdges.erase(new_end, dualEdges.end());
#endif

    // Sparse: every kept top cell has all four of its dual edges kept (a kept
    // top cell's faces are kept), so the dual graph reaches a pocket or the
    // exterior from everywhere and NOTHING survives -- every top-dim class
    // dies by CONE_BIRTH, because the box is acyclic. The scan over the kept
    // top cells (compact ids 0..K-1) is a safety net only.
    if (config.sparseComplex) {
        const index_t numTopCells =
            static_cast<index_t>(kept->keptTopCells.size());
        unordered_map<index_t, size_t> &essentialMatchMap =
            (k == 0) ? essentialMatchMap0 : essentialMatchMap1;
        for (index_t idx = 0; idx < numTopCells; ++idx) {
            if (uf.find(idx) == idx) {
                value_t rootBirth = uf.getBirth(idx);
                if (rootBirth != INFTY &&
                    rootBirth < config.maskThreshold) {
                    // Comp-root keyed (see src_3D/dimension_2.cpp).
                    if (rootBirth < TAU_REAL) {
                        essentialMatchMap.emplace(ufComp.find(idx),
                                                  essentials.size());
                    }
                    dim2::Coordinate coordinates = uf.getCoordinates(idx);
                    essentials.push_back(
                        Cube(rootBirth, std::get<0>(coordinates),
                             std::get<1>(coordinates), 0));
                }
            }
        }
    }

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms";
#endif
}

void Dimension1::computeCompPairsAndMatch(vector<Cube> &dualEdges) {
#ifdef RUNTIME
    cout << "barcode and matching ";
    auto start = high_resolution_clock::now();
#endif

    vector<index_t> boundaryIndices;
    index_t parentIdx0;
    index_t parentIdx1;
    index_t birthIdx;
    value_t birth;
    for (auto edge = dualEdges.rbegin(), last = dualEdges.rend(); edge != last;
         ++edge) {
        boundaryIndices = ufComp.getBoundaryIndices(*edge);
        parentIdx0 = ufComp.find(boundaryIndices[0]);
        parentIdx1 = ufComp.find(boundaryIndices[1]);
        if (parentIdx0 != parentIdx1) {
            birthIdx = ufComp.link(parentIdx0, parentIdx1);
            birth = ufComp.getBirth(birthIdx);
            if (edge->birth != birth) {
                if (ufComp.isPocketNode(birthIdx)) {
                    // Censored on the comparison side too: join the two input
                    // sides' censored records under the same comparison root,
                    // exactly as the finite path joins through matchMap0/1.
                    // A mixed case (censored on one side, finite on the other)
                    // cannot arise: a pocket sits at CONE_BIRTH, above every
                    // kept top cell, so it only ever dies against another
                    // pocket or the sentinel, in every one of the three UFs.
                    auto c0 = censoredMatchMap0.find(birthIdx);
                    auto c1 = censoredMatchMap1.find(birthIdx);
                    if (c0 != censoredMatchMap0.end() &&
                        c1 != censoredMatchMap1.end()) {
                        essentialMatches.push_back({c0->second, c1->second});
                    }
                    // NB: fall through to the clearing below -- the merge DID
                    // happen, so this edge is negative in dim 1 and must still
                    // be cleared from the worklist handed to dim 0.
                } else {
#ifdef COMPUTE_COMPARISON
                    auto birthCoordinates = ufComp.getCoordinates(birthIdx);
                    pairsComp.push_back(
                        Pair(*edge, Cube(birth, std::get<0>(birthCoordinates),
                                         std::get<1>(birthCoordinates), 0)));
#endif
                    auto find0 = matchMap0.find(birthIdx);
                    auto find1 = matchMap1.find(birthIdx);
                    if (find0 != matchMap0.end() && find1 != matchMap1.end()) {
                        matches.push_back(Match(find0->second, find1->second));
                        isMatched0.emplace(find0->second.birth.index, true);
                        isMatched1.emplace(find1->second.birth.index, true);
                        isMatchedWithIndexComp.emplace(edge->index,
                                                       matches.size() - 1);
                    }
                }
            }
#ifdef USE_CLEARING_DIM0
            edge->index = NONE_INDEX;
#endif
        }
    }

#ifdef USE_CLEARING_DIM0
    auto new_end = std::remove_if(
        dualEdges.begin(), dualEdges.end(),
        [](const Cube &cube) { return cube.index == NONE_INDEX; });
    dualEdges.erase(new_end, dualEdges.end());
#endif

    // Essential pass (top dimension, see src_3D/dimension_2.cpp); a safety
    // net only, nothing survives in sparse mode.
    if (config.sparseComplex) {
        const index_t numTopCells =
            static_cast<index_t>(kept->keptTopCells.size());
        for (index_t idx = 0; idx < numTopCells; ++idx) {
            if (ufComp.find(idx) != idx || ufComp.getBirth(idx) == INFTY) {
                continue;
            }
            index_t root0 = uf0.find(idx);
            index_t root1 = uf1.find(idx);
            if (uf0.getBirth(root0) >= TAU_REAL ||
                uf1.getBirth(root1) >= TAU_REAL) {
                continue;
            }
            auto find0 = essentialMatchMap0.find(idx);
            auto find1 = essentialMatchMap1.find(idx);
            if (find0 != essentialMatchMap0.end() &&
                find1 != essentialMatchMap1.end()) {
                essentialMatches.push_back({find0->second, find1->second});
            }
        }
    }

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms";
#endif
}

RepresentativeCycle
extractRepresentativeCycle(const Pair &pair, const CubicalGridComplex &cgc,
                           vector<dim2::Coordinate> &dualConnectedComponent) {
    map<Coordinate, size_t> boundaryVertices;
    for (const Coordinate &c : dualConnectedComponent) {
        for (uint8_t x = 0; x < 2; ++x) {
            for (uint8_t y = 0; y < 2; ++y) {
                auto vertex =
                    Coordinate{std::get<0>(c) + x, std::get<1>(c) + y};
                auto entry = boundaryVertices.find(vertex);
                if (entry == boundaryVertices.end()) {
                    boundaryVertices[vertex] = 1;
                } else {
                    entry->second++;
                }
            }
        }
    }

    vector<Coordinate> reprCycle;
    for (auto &entry : boundaryVertices) {
        if (entry.second < 4) {
            reprCycle.push_back(entry.first);
        }
    }
    reprCycle.push_back(cgc.getParentVoxel(pair.death, 2));

    return reprCycle;
}

vector<dim2::RepresentativeCycle> Dimension1::computeRepresentativeCycles(
    const int input,
    const std::vector<std::reference_wrapper<Pair>> &requestedPairs) {
    if (requestedPairs.size() == 0) {
        return {};
    }

    const CubicalGridComplex &cgc = (input == 0)   ? cgc0
                                    : (input == 1) ? cgc1
                                                   : cgcComp;
    UnionFindDual uf(cgc);

    vector<Cube> dualEdges;
    enumerateDualEdges(dualEdges, cgc);

    unordered_map<uint64_t, RepresentativeCycle> cyclesByBirth;

    // Initialize singleton dual connected components
    vector<RepresentativeCycle> dualConnectedComponentsByBirthIdx(cgc.shape[0] *
                                                                  cgc.shape[1]);
    for (int birthIdx = 0; birthIdx < dualConnectedComponentsByBirthIdx.size();
         birthIdx++) {
        dualConnectedComponentsByBirthIdx[birthIdx].emplace_back(
            uf.getCoordinates(birthIdx));
    }

    // Gather the birth indices of requested pairs for use in the loop below
    // (the pairs are not necessarily ordered in the same order as they are
    // found, hence we need a set)
    unordered_map<uint64_t, std::reference_wrapper<Pair>> requestedPairsByBirth;
    for (auto &pair : requestedPairs) {
        requestedPairsByBirth.emplace(pair.get().birth.index, pair);
    }

    // Run the union-find algorithm on the dual edges and collect all requested
    // representative cycles on the way
    for (auto edge = dualEdges.rbegin(), last = dualEdges.rend();
         edge != last && cyclesByBirth.size() != requestedPairs.size();
         edge++) {
        // Compute the representative cycle if it was requested
        vector<index_t> boundaryIndices = uf.getBoundaryIndices(*edge);
        index_t parentIdx0 = uf.find(boundaryIndices[0]);
        index_t parentIdx1 = uf.find(boundaryIndices[1]);
        if (parentIdx0 != parentIdx1) {
            auto olderBirthIdx = uf.link(parentIdx0, parentIdx1);
            auto youngerBirthIdx =
                (parentIdx1 == olderBirthIdx) ? parentIdx0 : parentIdx1;
            dualConnectedComponentsByBirthIdx[youngerBirthIdx].insert(
                dualConnectedComponentsByBirthIdx[youngerBirthIdx].end(),
                dualConnectedComponentsByBirthIdx[olderBirthIdx].begin(),
                dualConnectedComponentsByBirthIdx[olderBirthIdx].end());

            auto maybeRequestedPair = requestedPairsByBirth.find(edge->index);
            if (maybeRequestedPair != requestedPairsByBirth.end()) {
                cyclesByBirth[edge->index] = extractRepresentativeCycle(
                    maybeRequestedPair->second, cgc,
                    dualConnectedComponentsByBirthIdx[olderBirthIdx]);
            }
            // The deceased dual connected component has either been converted
            // to its bordering 2-cycle, or the 2-cycle has not been requested.
            // Either way, we can delete it and save memory.
            dualConnectedComponentsByBirthIdx[olderBirthIdx].clear();
        }
    }

    if (cyclesByBirth.size() != requestedPairs.size()) {
        throw runtime_error(
            "Not all requested representative cycles were found");
    }

    vector<RepresentativeCycle> representativeCycles;
    representativeCycles.reserve(requestedPairs.size());
    for (auto &pair : requestedPairs) {
        representativeCycles.emplace_back(
            std::move(cyclesByBirth[pair.get().birth.index]));
    }
    return representativeCycles;
}
