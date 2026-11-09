// Strata (this fork). The speed-aware layer split of strata-dsv41 (src/dsv41/ds4_split_speed.hpp, DS4_SPLIT_SPEED): pure, no GPU.
#include "ds4_split_speed.hpp"

#include <cstdio>
#include <vector>

using ds4split::LayerBytes;
using ds4split::Model;

static int fails = 0;
#define CHECK(c)                                                  \
    do {                                                          \
        if (!(c)) {                                               \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                              \
        }                                                         \
    } while (0)

static const double GB = 1e9;

// 40 layers shaped like DeepSeek-V4.1-Flash at 2 bits: ~0.42 GB of dense weights and ~3.3 GB of experts a layer, a token reading 6 experts
static std::vector<LayerBytes> layers(int n = 40) {
    std::vector<LayerBytes> v((size_t) n);
    for (auto& l : v) l = {0.45 * GB, 3.3 * GB, 0.42 * GB, 6 * 3.3 * GB / 384};
    return v;
}

// the memory split's shape: contiguous, each group's share by its room
static std::vector<int> by_room(const std::vector<double>& room, int n) {
    double sum = 0;
    for (double r : room) sum += r;
    std::vector<int> g((size_t) n);
    double cum = 0, target = room[0] / sum * n;
    int d = 0;
    for (int i = 0; i < n; ++i) {
        if (cum + 0.5 > target && d + 1 < (int) room.size()) target += room[(size_t) ++d] / sum * n;
        g[(size_t) i] = d;
        cum += 1;
    }
    return g;
}

static int count(const std::vector<int>& g, int k) {
    int n = 0;
    for (int v : g) n += v == k;
    return n;
}

int main() {
    const auto L = layers();
    {   // two equal cards, the CPU slower: nothing beats the even memory split (splits that move the same experts to the lane tie)
        const std::vector<double> room = {24 * GB, 24 * GB};
        const Model m(L, room, {600, 600}, 20);
        const auto r = m.best(by_room(room, 40));
        CHECK(r.ok);
        CHECK(r.t_after >= r.t_before - 1e-12 && r.t_after <= r.t_before + 1e-12);
    }
    {   // the rig of the question: 3090 24 GB, V100 32 GB, M6000 24 GB, M6000 12 GB; the M6000s far slower on the dense work
        const std::vector<double> room = {22 * GB, 30 * GB, 22 * GB, 10.5 * GB};
        const std::vector<double> gbs = {600, 500, 120, 120};
        const auto before = by_room(room, 40);
        const Model m(L, room, gbs, 25);
        const auto r = m.best(before);
        CHECK(r.ok);
        CHECK(r.t_after <= r.t_before);                         // never worse than the memory split by its own estimate
        CHECK(count(r.grp, 2) + count(r.grp, 3) <= count(before, 2) + count(before, 3));   // the slow cards do not get more layers
        for (int i = 1; i < 40; ++i) CHECK(r.grp[(size_t) i] >= r.grp[(size_t) i - 1]);  // contiguous, in card order
        for (int g = 0; g < 4; ++g) CHECK(count(r.grp, g) >= 1);
        std::printf("4-card example: layers %d,%d,%d,%d (memory split %d,%d,%d,%d), estimate %.1f ms against %.1f ms\n", count(r.grp, 0),
                    count(r.grp, 1), count(r.grp, 2), count(r.grp, 3), count(before, 0), count(before, 1), count(before, 2),
                    count(before, 3), r.t_after * 1e3, r.t_before * 1e3);
    }
    {   // the same rig with RAM in the estimate: experts past RAM come from the SSD at 3 GB/s.  Little RAM keeps more of them in the slow
        // cards' VRAM than plenty of RAM does
        const std::vector<double> room = {22 * GB, 30 * GB, 22 * GB, 10.5 * GB};
        const std::vector<double> gbs = {600, 500, 120, 120};
        const auto before = by_room(room, 40);
        const Model small_ram(L, room, gbs, 25, 24 * GB, 3), big_ram(L, room, gbs, 25, 120 * GB, 3);
        const auto rs = small_ram.best(before), rb = big_ram.best(before);
        CHECK(rs.ok && rb.ok);
        CHECK(rs.t_after <= rs.t_before && rb.t_after <= rb.t_before);
        CHECK(count(rs.grp, 2) + count(rs.grp, 3) >= count(rb.grp, 2) + count(rb.grp, 3));
        CHECK(small_ram.lane_gbs(before) < 25);   // the memory split already reads part of its lane from the SSD with 24 GB
        std::printf("with 24 GB of RAM: layers %d,%d,%d,%d, %.1f ms against %.1f; with 120 GB: %d,%d,%d,%d, %.1f ms against %.1f\n",
                    count(rs.grp, 0), count(rs.grp, 1), count(rs.grp, 2), count(rs.grp, 3), rs.t_after * 1e3, rs.t_before * 1e3,
                    count(rb.grp, 0), count(rb.grp, 1), count(rb.grp, 2), count(rb.grp, 3), rb.t_after * 1e3, rb.t_before * 1e3);
    }
    {   // every layer's dense weights fit only if spread: a card's room below one layer's dense weights has no split
        const std::vector<double> room = {0.3 * GB, 40 * GB};
        const Model m(L, room, {600, 600}, 20);
        CHECK(!m.best(by_room(room, 40)).ok);
    }
    {   // a model that fits in VRAM: the faster card takes as many layers as its memory lets, the slower one the rest
        const auto small = layers(10);
        const std::vector<double> room = {40 * GB, 40 * GB};
        const Model m(small, room, {800, 100}, 20);
        const auto r = m.best(by_room(room, 10));
        CHECK(r.ok);
        CHECK(count(r.grp, 0) >= 9);   // 9 x 3.75 GB fit in 40 GB; the second card still needs one layer
        CHECK(r.t_after < r.t_before);
    }
    {   // equal speed everywhere, the CPU as fast as the cards: the estimate cannot improve on the memory split by much and never worsens it
        const std::vector<double> room = {22 * GB, 30 * GB};
        const Model m(L, room, {300, 300}, 300);
        const auto r = m.best(by_room(room, 40));
        CHECK(r.ok && r.t_after <= r.t_before + 1e-12);
    }
    if (fails == 0) std::printf("split_speed_test: ok\n");
    return fails == 0 ? 0 : 1;
}
