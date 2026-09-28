#include "BettiMatching.h"
#include "../utils.h"
#include "data_structures.h"
#include "dimension_0.h"
#include "dimension_1.h"
#include "dimension_2.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>

using namespace dim3;
using namespace std;
using namespace std::chrono;

BettiMatching::BettiMatching(vector<value_t> &&input0, vector<value_t> &&input1,
                             vector<value_t> &&comparison,
                             vector<index_t> &&shape, Config &&_config)
    : cgc0(input0, shape), cgc1(input1, shape), cgcComp(comparison, shape),
      config(_config) {
    if (config.sparseComplex) {
        keptCells = std::make_unique<KeptCells>(cgcComp);
        if (config.pocketLabels == nullptr) {
            throw runtime_error("sparse=True requires pocket_labels");
        }
        pocketNodes = std::make_unique<PocketNodes>(
            cgcComp, *config.pocketLabels,
            keptCells->keptVertices.size());
#ifdef RUNTIME
        auto _tcl = std::chrono::high_resolution_clock::now();
#endif
        for (const index_t v : keptCells->keptVertices) {
            const index_t x = v / (cgcComp.shape[1] * cgcComp.shape[2]);
            const index_t y = (v / cgcComp.shape[2]) % cgcComp.shape[1];
            const index_t z = v % cgcComp.shape[2];
            if (cgc0.getBirth(x, y, z) > CONE_BIRTH ||
                cgc1.getBirth(x, y, z) > CONE_BIRTH) {
                throw runtime_error(
                    "sparse=True requires all kept values to be at most "
                    "1.0 (inputs are g = 1 - p in [0, 1])");
            }
        }
#ifdef RUNTIME
        g_tRangeCheck += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - _tcl).count();
#endif
        coneIndex = std::make_unique<ConeIndex>(
            cgcComp, *keptCells, *pocketNodes, config.conePrune);
    }
    pairs0 = vector<vector<Pair>>(3);
    pairs1 = vector<vector<Pair>>(3);
    pairsComp = vector<vector<Pair>>(3);
    matches = vector<vector<Match>>(3);
    isMatched0 = vector<unordered_map<uint64_t, bool>>(3);
    isMatched1 = vector<unordered_map<uint64_t, bool>>(3);
    isMatchedWithIndexComp = vector<unordered_map<uint64_t, size_t>>(3);
    _matched = vector<vector<VoxelMatch>>(3);
    _unmatched0 = vector<vector<VoxelPair>>(3);
    _unmatched1 = vector<vector<VoxelPair>>(3);
    essentials0 = vector<vector<Cube>>(3);
    essentials1 = vector<vector<Cube>>(3);
    essentialMatches = vector<vector<pair<size_t, size_t>>>(3);
    _unmatchedEssentials0 = vector<vector<vector<index_t>>>(3);
    _unmatchedEssentials1 = vector<vector<vector<index_t>>>(3);
    _matchedEssentials0 = vector<vector<vector<index_t>>>(3);
    _matchedEssentials1 = vector<vector<vector<index_t>>>(3);
    _matchedComp = vector<vector<VoxelPair>>(3);
#ifdef RUNTIME
    cerr << "INIT_BREAKDOWN"
         << " grid3=" << g_tGrid * 1000 << " kept=" << g_tKept * 1000
         << " pocket_validate=" << g_tPocketVal * 1000
         << " pocket_topcell=" << g_tPocketTop * 1000
         << " rangecheck=" << g_tRangeCheck * 1000 << " cone_VE=" << g_tConeVE * 1000
         << " cone_faces=" << g_tConeFaces * 1000
         << " cone_incidence=" << g_tConeInc * 1000
         << " cone_emitmask=" << g_tConeEmit * 1000
         << " cone_cycles=" << g_tConeCycles * 1000 << endl;
    g_tGrid = g_tKept = g_tPocketVal = g_tPocketTop =
        g_tRangeCheck = g_tConeVE = g_tConeFaces = g_tConeInc = g_tConeEmit =
            g_tConeCycles = 0;
#endif
}

BettiMatching::BettiMatching(BettiMatching &&other)
    : cgc0(std::move(other.cgc0)), cgc1(std::move(other.cgc1)),
      cgcComp(std::move(other.cgcComp)), config(other.config),
      keptCells(std::move(other.keptCells)),
      pocketNodes(std::move(other.pocketNodes)),
      coneIndex(std::move(other.coneIndex)),
      pairs0(std::move(other.pairs0)), pairs1(std::move(other.pairs1)),
      pairsComp(std::move(other.pairsComp)),
      matches(std::move(other.matches)),
      isMatched0(std::move(other.isMatched0)),
      isMatched1(std::move(other.isMatched1)),
      isMatchedWithIndexComp(std::move(other.isMatchedWithIndexComp)),
      essentials0(std::move(other.essentials0)),
      essentials1(std::move(other.essentials1)),
      essentialMatches(std::move(other.essentialMatches)),
      _matched(std::move(other._matched)),
      _unmatched0(std::move(other._unmatched0)),
      _unmatched1(std::move(other._unmatched1)),
      _unmatchedEssentials0(std::move(other._unmatchedEssentials0)),
      _unmatchedEssentials1(std::move(other._unmatchedEssentials1)),
      _matchedEssentials0(std::move(other._matchedEssentials0)),
      _matchedEssentials1(std::move(other._matchedEssentials1)),
      _matchedComp(std::move(other._matchedComp))
#ifdef USE_CACHE
      ,
      dim1CacheInputPairs0(std::move(other.dim1CacheInputPairs0)),
      dim1CacheInputPairs1(std::move(other.dim1CacheInputPairs1)),
      dim1CacheCompPairs(std::move(other.dim1CacheCompPairs))
#endif
{
}

void BettiMatching::computeMatching() {
    vector<Cube> ctr0;
    vector<Cube> ctr1;
    vector<Cube> ctrComp;
    vector<Cube> ctrImage;

    {
#ifdef RUNTIME
        cout << "dimension 2:";
        auto start = high_resolution_clock::now();
#endif

        Dimension2 dim2(cgc0, cgc1, cgcComp, config, pairs0[2], pairs1[2],
                        pairsComp[2], matches[2], isMatched0[2], isMatched1[2],
                        isMatchedWithIndexComp[2], essentials0[2],
                        essentials1[2], essentialMatches[2], keptCells.get(),
                        pocketNodes.get());
        dim2.computePairsAndMatch(ctr0, ctr1, ctrComp, ctrImage);

#ifdef RUNTIME
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << endl << "total: " << duration.count() << " ms" << endl << endl;
#endif
    }

    {
#ifdef RUNTIME
        cout << "dimension 1:";
        auto start = high_resolution_clock::now();
#endif

        dim1CacheInputPairs0.emplace(cgc0.shape, keptCells.get(), coneIndex.get());
        dim1CacheInputPairs1.emplace(cgc0.shape, keptCells.get(), coneIndex.get());
        dim1CacheCompPairs.emplace(cgc0.shape, keptCells.get(), coneIndex.get());
        Dimension1 dim1(cgc0, cgc1, cgcComp, config, pairs0[1], pairs1[1],
                        pairsComp[1], matches[1], isMatched0[1], isMatched1[1],
                        isMatchedWithIndexComp[1], essentials0[1], essentials1[1],
                        essentialMatches[1], keptCells.get(), coneIndex.get(),
                        *dim1CacheInputPairs0, *dim1CacheInputPairs1,
                        *dim1CacheCompPairs);
        dim1.computePairsAndMatch(ctr0, ctr1, ctrComp, ctrImage);

#ifdef RUNTIME
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << endl << "total: " << duration.count() << " ms" << endl << endl;
#endif
    }

    {
#ifdef RUNTIME
        cout << "dimension 0:";
        auto start = high_resolution_clock::now();
#endif

        Dimension0 dim0(cgc0, cgc1, cgcComp, config, pairs0[0], pairs1[0],
                        pairsComp[0], matches[0], isMatched0[0], isMatched1[0],
                        isMatchedWithIndexComp[0], essentials0[0],
                        essentials1[0], essentialMatches[0], keptCells.get());
        dim0.computePairsAndMatch(ctr0, ctr1, ctrComp);

#ifdef RUNTIME
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << endl << "total: " << duration.count() << " ms" << endl << endl;
#endif
    }

#ifdef USE_CACHE
    if (config.releaseCachesAfterMatching) {
        dim1CacheInputPairs0.reset();
        dim1CacheInputPairs1.reset();
        dim1CacheCompPairs.reset();
    }
#endif
}

void BettiMatching::computeVoxels() {
#ifdef RUNTIME
    cout << "computing voxels ... ";
    auto start = high_resolution_clock::now();
#endif

    for (uint8_t d = 0; d < 3; ++d) {
        for (auto &pair : pairs0[d]) {
            if (!isMatched0[d][pair.birth.index]) {
                _unmatched0[d].push_back(
                    VoxelPair(cgc0.getParentVoxel(pair.birth, d),
                              cgc0.getParentVoxel(pair.death, d + 1)));
            }
        }
        for (auto &pair : pairs1[d]) {
            if (!isMatched1[d][pair.birth.index]) {
                _unmatched1[d].push_back(
                    VoxelPair(cgc1.getParentVoxel(pair.birth, d),
                              cgc1.getParentVoxel(pair.death, d + 1)));
            }
        }
        for (auto &match : matches[d]) {
            _matched[d].push_back(VoxelMatch(
                VoxelPair(cgc0.getParentVoxel(match.pair0.birth, d),
                          cgc0.getParentVoxel(match.pair0.death, d + 1)),
                VoxelPair(cgc1.getParentVoxel(match.pair1.birth, d),
                          cgc1.getParentVoxel(match.pair1.death, d + 1))));
        }
        vector<const Pair *> compByMatch(matches[d].size(), nullptr);
        for (auto &pair : pairsComp[d]) {
            auto find = isMatchedWithIndexComp[d].find(pair.birth.index);
            if (find != isMatchedWithIndexComp[d].end()) {
                compByMatch[find->second] = &pair;
            }
        }
        for (const Pair *compPair : compByMatch) {
            if (compPair == nullptr) {
                throw runtime_error(
                    "match without a comparison pair in dimension " +
                    std::to_string(d));
            }
            _matchedComp[d].push_back(VoxelPair(
                cgcComp.getParentVoxel(compPair->birth, d),
                cgcComp.getParentVoxel(compPair->death, d + 1)));
        }

        const uint8_t cellDim = d;
        vector<bool> matched0(essentials0[d].size(), false);
        vector<bool> matched1(essentials1[d].size(), false);
        for (auto &match : essentialMatches[d]) {
            matched0[match.first] = true;
            matched1[match.second] = true;
            _matchedEssentials0[d].push_back(tupleToVector(
                cgc0.getParentVoxel(essentials0[d][match.first], cellDim)));
            _matchedEssentials1[d].push_back(tupleToVector(
                cgc1.getParentVoxel(essentials1[d][match.second], cellDim)));
        }
        for (size_t i = 0; i < essentials0[d].size(); ++i) {
            if (!matched0[i] && essentials0[d][i].birth < TAU_REAL) {
                _unmatchedEssentials0[d].push_back(tupleToVector(
                    cgc0.getParentVoxel(essentials0[d][i], cellDim)));
            }
        }
        for (size_t i = 0; i < essentials1[d].size(); ++i) {
            if (!matched1[i] && essentials1[d][i].birth < TAU_REAL) {
                _unmatchedEssentials1[d].push_back(tupleToVector(
                    cgc1.getParentVoxel(essentials1[d][i], cellDim)));
            }
        }
    }

#ifdef RUNTIME
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl << endl;
#endif
}

vector<vector<VoxelPair>> BettiMatching::computePairsInput0() {
    vector<Cube> ctr0;

    Dimension2 dim2(cgc0, cgc1, cgcComp, config, pairs0[2], pairs1[2],
                    pairsComp[2], matches[2], isMatched0[2], isMatched1[2],
                    isMatchedWithIndexComp[2], essentials0[2], essentials1[2],
                    essentialMatches[2], keptCells.get(), pocketNodes.get());
    dim2.computeInput0Pairs(ctr0);

    dim1CacheInputPairs0.emplace(cgc0.shape, keptCells.get(), coneIndex.get());
    dim1CacheInputPairs1.emplace(vector<index_t>{1, 1, 1});
    dim1CacheCompPairs.emplace(vector<index_t>{1, 1, 1});
    Dimension1 dim1(cgc0, cgc1, cgcComp, config, pairs0[1], pairs1[1],
                    pairsComp[1], matches[1], isMatched0[1], isMatched1[1],
                    isMatchedWithIndexComp[1], essentials0[1], essentials1[1],
                    essentialMatches[1], keptCells.get(), coneIndex.get(),
                    *dim1CacheInputPairs0, *dim1CacheInputPairs1,
                    *dim1CacheCompPairs);
    dim1.computeInput0Pairs(ctr0);

    Dimension0 dim0(cgc0, cgc1, cgcComp, config, pairs0[0], pairs1[0],
                    pairsComp[0], matches[0], isMatched0[0], isMatched1[0],
                    isMatchedWithIndexComp[0], essentials0[0], essentials1[0],
                    essentialMatches[0], keptCells.get());
    dim0.computeInput0Pairs(ctr0);

    vector<vector<VoxelPair>> voxelPairs(3);

    for (uint8_t d = 0; d < 3; ++d) {
        for (auto &pair : pairs0[d]) {
            voxelPairs[d].push_back(
                VoxelPair(cgc0.getParentVoxel(pair.birth, d),
                          cgc0.getParentVoxel(pair.death, d + 1)));
        }
    }
    return voxelPairs;
}

void BettiMatching::printResult() {
    index_t count;

    cout << "------------------------------------------------------------------"
            "---------------------------------------------"
         << endl;
    cout << "Input 0:" << endl;
    if (cgc0.shape[0] < 10 && cgc0.shape[1] < 10 && cgc0.shape[2] < 10) {
        cgc0.printImage();
    }
    for (uint8_t d = 0; d < 3; ++d) {
        cout << "dim " << unsigned(d) << ": ";
        count = pairs0[d].size();
        if (0 < count && count < 10) {
            cout << endl;
            for (auto &pair : pairs0[d]) {
                pair.print();
                cout << endl;
            }
        } else {
            cout << count << endl;
        }
    }

    cout << "------------------------------------------------------------------"
            "---------------------------------------------"
         << endl;
    cout << "Input 1:" << endl;
    if (cgc1.shape[0] < 10 && cgc1.shape[1] < 10 && cgc1.shape[2] < 10) {
        cgc1.printImage();
    }
    for (uint8_t d = 0; d < 3; ++d) {
        cout << "dim " << unsigned(d) << ": ";
        count = pairs1[d].size();
        if (0 < count && count < 10) {
            cout << endl;
            for (auto &pair : pairs1[d]) {
                pair.print();
                cout << endl;
            }
        } else {
            cout << count << endl;
        }
    }

#ifdef COMPUTE_COMPARISON
    cout << "------------------------------------------------------------------"
            "---------------------------------------------"
         << endl;
    cout << "Comparison:" << endl;
    if (cgcComp.shape[0] < 10 && cgcComp.shape[1] < 10 &&
        cgcComp.shape[2] < 10) {
        cgcComp.printImage();
    }
    for (uint8_t d = 0; d < 3; ++d) {
        cout << "dim " << unsigned(d) << ": ";
        count = pairsComp[d].size();
        if (0 < count && count < 10) {
            cout << endl;
            for (auto &pair : pairsComp[d]) {
                pair.print();
                cout << endl;
            }
        } else {
            cout << count << endl;
        }
    }
#endif

    cout << "------------------------------------------------------------------"
            "---------------------------------------------"
         << endl;
    cout << "Betti Matching:" << endl << endl;
    cout << "matched: " << endl;
    for (uint8_t d = 0; d < 3; ++d) {
        cout << "dim " << unsigned(d) << ": ";
        count = matches[d].size();
        if (0 < count && count < 10) {
            cout << endl;
            std::vector<size_t> range(count);
            std::iota(range.begin(), range.end(), 0);
            vector<RepresentativeCycle> matchedCycles0 =
                std::get<0>(computeRepresentativeCycles(
                    0, d, range, {std::vector<size_t>()}));
            vector<RepresentativeCycle> matchedCycles1 =
                std::get<0>(computeRepresentativeCycles(
                    1, d, range, {std::vector<size_t>()}));
            for (size_t i = 0; i < matches[d].size(); i++) {
                matches[d][i].print();
                _matched[d][i].print();
                if (cgc0.shape[0] < 10 && cgc0.shape[1] < 10 &&
                    cgc0.shape[2] < 10) {
                    cgc0.printRepresentativeCycle(matchedCycles0[i]);
                    cout << endl;
                    cgc1.printRepresentativeCycle(matchedCycles1[i]);
                    cout << endl;
                }
            }
        } else {
            cout << count << endl;
        }
    }

    size_t counter;
    cout << endl << "unmatched in Input 0:" << endl;
    for (uint8_t d = 0; d < 3; ++d) {
        cout << "dim " << unsigned(d) << ": ";
        count = 0;
        for (auto &pair : pairs0[d]) {
            if (!isMatched0[d][pair.birth.index]) {
                ++count;
            }
        }
        if (0 < count && count < 10) {
            cout << endl;
            std::vector<size_t> range(count);
            std::iota(range.begin(), range.end(), 0);
            vector<RepresentativeCycle> unmatchedCycles =
                std::get<1>(computeRepresentativeCycles(0, d, {std::vector<size_t>()}, range));
            counter = 0;
            for (size_t i = 0; i < pairs0[d].size(); ++i) {
                if (!isMatched0[d][pairs0[d][i].birth.index]) {
                    pairs0[d][i].print();
                    cout << endl;
                    _unmatched0[d][counter].print();
                    cout << endl;
                    cgc0.printRepresentativeCycle(unmatchedCycles[counter]);
                    cout << endl;
                    ++counter;
                    if (counter == count) {
                        break;
                    }
                }
            }
        } else {
            cout << count << endl;
        }
    }

    cout << endl << "unmatched in Input 1:" << endl;
    for (uint8_t d = 0; d < 3; ++d) {
        cout << "dim " << unsigned(d) << ": ";
        count = 0;
        for (auto &pair : pairs1[d]) {
            if (!isMatched1[d][pair.birth.index]) {
                ++count;
            }
        }
        if (0 < count && count < 10) {
            cout << endl;
            std::vector<size_t> range(count);
            std::iota(range.begin(), range.end(), 0);
            vector<RepresentativeCycle> unmatchedCycles =
                std::get<1>(computeRepresentativeCycles(
                    1, d, {std::vector<size_t>()}, range));
            counter = 0;
            for (size_t i = 0; i < pairs1[d].size(); ++i) {
                if (!isMatched1[d][pairs1[d][i].birth.index]) {
                    pairs1[d][i].print();
                    cout << endl;
                    _unmatched1[d][counter].print();
                    cout << endl;
                    cgc1.printRepresentativeCycle(unmatchedCycles[counter]);
                    cout << endl;
                    ++counter;
                    if (counter == count) {
                        break;
                    }
                }
            }
        } else {
            cout << count << endl;
        }
    }
}

tuple<vector<dim3::RepresentativeCycle>, vector<dim3::RepresentativeCycle>>
BettiMatching::computeRepresentativeCycles(
    const int input, const int dim,
    const optional<vector<size_t>> &matchedPairsIndices,
    const optional<vector<size_t>> &unmatchedPairsIndices) {
    if (dim < 0 || dim > 2) {
        throw runtime_error("Invalid value for dim");
    }

    // Assemble the list of requested pairs: First the matched pairs (all if
    // empty optional was passed, then the unmatched pairs (all if empty
    // optional was passed)
    vector<std::reference_wrapper<Pair>> requestedPairs;
    if (input == 0 || input == 1) {
        requestedPairs = assembleRequestedPairs(
            matchedPairsIndices, unmatchedPairsIndices,
            (input == 0 ? pairs0 : pairs1)[dim],
            (input == 0 ? isMatched0 : isMatched1)[dim], matches[dim], input);
    } else if (input == 2) {
        requestedPairs = assembleRequestedComparisonPairs<Pair>(
            matchedPairsIndices, pairsComp[dim], isMatchedWithIndexComp[dim]);
    } else {
        throw runtime_error("Invalid value for input");
    }

    // Hand over the representative cycle computation to the respective
    // dimension
    vector<RepresentativeCycle> representativeCycles;
    switch (dim) {
    case 0: {
        Dimension0 dim0(cgc0, cgc1, cgcComp, config, pairs0[0], pairs1[0],
                        pairsComp[0], matches[0], isMatched0[0], isMatched1[0],
                        isMatchedWithIndexComp[0], essentials0[0],
                        essentials1[0], essentialMatches[0], keptCells.get());
        representativeCycles =
            dim0.computeRepresentativeCycles(input, requestedPairs);
        break;
    }
    case 1: {
        if (!dim1CacheInputPairs0 || !dim1CacheInputPairs1 ||
            !dim1CacheCompPairs) {
            throw runtime_error(
                "Dim-1 representative cycles need the reduction caches, but "
                "they were released (release_caches_after_matching=True) or "
                "matching has not been computed yet");
        }
        Dimension1 dim1(cgc0, cgc1, cgcComp, config, pairs0[1], pairs1[1],
                        pairsComp[1], matches[1], isMatched0[1], isMatched1[1],
                        isMatchedWithIndexComp[1], essentials0[1], essentials1[1],
                        essentialMatches[1], keptCells.get(), coneIndex.get(),
                        *dim1CacheInputPairs0, *dim1CacheInputPairs1,
                        *dim1CacheCompPairs);
        representativeCycles =
            dim1.computeRepresentativeCycles(input, requestedPairs);
        break;
    }
    case 2: {
        Dimension2 dim2(cgc0, cgc1, cgcComp, config, pairs0[2], pairs1[2],
                        pairsComp[2], matches[2], isMatched0[2], isMatched1[2],
                        isMatchedWithIndexComp[2], essentials0[2],
                        essentials1[2], essentialMatches[2], keptCells.get(),
                        pocketNodes.get());
        representativeCycles =
            dim2.computeRepresentativeCycles(input, requestedPairs);
        break;
    }
    }

    // Split the returned representative cycles into a matched and unmatched
    // portion
    vector<RepresentativeCycle> matchedRepresentativeCycles;
    vector<RepresentativeCycle> unmatchedRepresentativeCycles;

    int numMatchedRequested = matchedPairsIndices.has_value()
                                  ? matchedPairsIndices->size()
                                  : matches[dim].size();
    matchedRepresentativeCycles.insert(
        matchedRepresentativeCycles.end(),
        std::make_move_iterator(representativeCycles.begin()),
        std::make_move_iterator(representativeCycles.begin()) +
            numMatchedRequested);
    unmatchedRepresentativeCycles.insert(
        unmatchedRepresentativeCycles.end(),
        std::make_move_iterator(representativeCycles.begin() +
                                numMatchedRequested),
        std::make_move_iterator(representativeCycles.end()));

    return {matchedRepresentativeCycles, unmatchedRepresentativeCycles};
}
