// DS4_SPLIT_SPEED (this fork): the contiguous layer split of strata-dsv41 by each card's speed as well as its memory.  Pure functions, no
// GPU: ds4_model.cu fills the layers and the groups' room, ds4_split_speed_test.cpp checks the choice.
//
// A token's decode passes every trunk layer once, one group after the other, so the time of a split is the SUM over the groups of
//     the dense weights a token reads / s_g  +  max(the experts' share in the group's VRAM / s_g,  the share left to the CPU lane / s_cpu)
// (the CPU lane runs beside the card's own experts and the card waits for it).  A group's spare VRAM, after the dense weights of its
// layers, is spread over its layers' experts; an expert in VRAM is counted as hit in proportion to their share there (the engine keeps
// the hottest ones in VRAM, so this undercounts the hits).  Speeds are GB/s on the dense work (the DS4_PROF readout prints them).
// The CPU lane's experts beyond what RAM can cache (`ram`) come from the SSD: their share of the lane's reads runs at `disk_gbs`.  That
// couples the groups, so the split is found with the lane's speed held fixed, the speed worked out again from the split, a few rounds,
// and the best split by the whole estimate kept.
#pragma once

#include <algorithm>
#include <vector>

namespace ds4split {

struct LayerBytes {
    double held_dense = 0;     // dense weights + rings it keeps in VRAM
    double held_experts = 0;   // all its routed experts (n_exp x blob)
    double read_dense = 0;     // dense weights a token reads
    double read_experts = 0;   // expert bytes a token reads (k_exp x blob)
};

struct Result {
    bool ok = false;               // a split with every group's dense weights in its VRAM exists
    std::vector<int> grp;          // the group of each layer (contiguous, every group at least one layer)
    double t_before = 0, t_after = 0;   // seconds a token by the estimate: the split given, the one chosen
};

class Model {
public:
    Model(const std::vector<LayerBytes>& layers, const std::vector<double>& room, const std::vector<double>& gbs, double cpu_gbs,
          double ram = 1e300, double disk_gbs = 3.0)
        : room_(room), gbs_(gbs), cpu_(cpu_gbs), ram_(ram), disk_(disk_gbs) {
        const size_t n = layers.size();
        pd_.assign(n + 1, 0.0);
        pe_ = pw_ = px_ = pd_;
        for (size_t i = 0; i < n; ++i) {
            pd_[i + 1] = pd_[i] + layers[i].held_dense;
            pe_[i + 1] = pe_[i] + layers[i].held_experts;
            pw_[i + 1] = pw_[i] + layers[i].read_dense;
            px_[i + 1] = px_[i] + layers[i].read_experts;
        }
    }
    int layers() const { return (int) pd_.size() - 1; }
    int groups() const { return (int) room_.size(); }

    /// Seconds a token spends in layers [a, b) on group g with the CPU lane at `cpu` GB/s; 1e300 when their dense weights do not fit.
    double cost(int g, int a, int b, double cpu) const {
        const double dense = pd_[(size_t) b] - pd_[(size_t) a], ex = pe_[(size_t) b] - pe_[(size_t) a];
        if (dense > room_[(size_t) g]) return 1e300;
        const double f = ex > 0 ? std::min(1.0, (room_[(size_t) g] - dense) / ex) : 1.0;
        const double sg = gbs_[(size_t) g] * 1e9, sc = cpu * 1e9, x = px_[(size_t) b] - px_[(size_t) a];
        return (pw_[(size_t) b] - pw_[(size_t) a]) / sg + std::max(x * f / sg, x * (1.0 - f) / sc);
    }

    /// The experts an assignment leaves to the CPU lane (bytes held), and the lane's speed then: the RAM's share at cpu_gbs, the rest at disk_gbs.
    double lane_gbs(const std::vector<int>& grp) const {
        const int n = layers();
        double left = 0;
        for (int a = 0; a < n;) {
            int b = a;
            while (b < n && grp[(size_t) b] == grp[(size_t) a]) ++b;
            const double dense = pd_[(size_t) b] - pd_[(size_t) a], ex = pe_[(size_t) b] - pe_[(size_t) a];
            left += std::max(0.0, ex - std::max(0.0, room_[(size_t) grp[(size_t) a]] - dense));
            a = b;
        }
        if (left <= ram_ || left <= 0) return cpu_;
        const double d = (left - ram_) / left;   // the share of the lane's reads that miss RAM
        return 1.0 / ((1.0 - d) / cpu_ + d / disk_);
    }

    /// The estimate for a whole assignment (contiguous runs of one group).
    double total(const std::vector<int>& grp) const {
        const double cpu = lane_gbs(grp);
        double t = 0;
        const int n = layers();
        for (int a = 0; a < n;) {
            int b = a;
            while (b < n && grp[(size_t) b] == grp[(size_t) a]) ++b;
            const double c = cost(grp[(size_t) a], a, b, cpu);
            if (c >= 1e299) return 1e300;
            t += c;
            a = b;
        }
        return t;
    }

    /// The contiguous split (groups in order, each at least one layer) with the least total; `before` is the split it is compared with.
    Result best(const std::vector<int>& before) const {
        Result r;
        r.t_before = total(before);
        double cpu = r.t_before < 1e299 ? lane_gbs(before) : cpu_;
        for (int round = 0; round < 6; ++round) {
            std::vector<int> grp;
            if (!split_at(cpu, grp)) break;
            const double t = total(grp);
            if (!r.ok || t < r.t_after) {
                r.ok = true;
                r.grp = grp;
                r.t_after = t;
            }
            const double next = lane_gbs(grp);
            if (next == cpu) break;
            cpu = next;
        }
        return r;
    }

private:
    bool split_at(double cpu, std::vector<int>& out) const {
        const int G = groups(), N = layers();
        if (G < 1 || N < G) return false;
        std::vector<std::vector<double>> bst((size_t) G, std::vector<double>((size_t) N + 1, 1e300));
        std::vector<std::vector<int>> cut((size_t) G, std::vector<int>((size_t) N + 1, 0));
        for (int b = 1; b <= N; ++b) bst[0][(size_t) b] = cost(0, 0, b, cpu);
        for (int g = 1; g < G; ++g)
            for (int b = g + 1; b <= N; ++b)
                for (int a = g; a < b; ++a) {
                    const double p = bst[(size_t) g - 1][(size_t) a];
                    if (p >= 1e299) continue;
                    const double c = cost(g, a, b, cpu);
                    if (c >= 1e299) continue;
                    if (p + c < bst[(size_t) g][(size_t) b]) {
                        bst[(size_t) g][(size_t) b] = p + c;
                        cut[(size_t) g][(size_t) b] = a;
                    }
                }
        if (bst[(size_t) G - 1][(size_t) N] >= 1e299) return false;
        out.assign((size_t) N, 0);
        int b = N;
        for (int g = G - 1; g >= 0; --g) {
            const int a = g > 0 ? cut[(size_t) g][(size_t) b] : 0;
            for (int i = a; i < b; ++i) out[(size_t) i] = g;
            b = a;
        }
        return true;
    }

    std::vector<double> room_, gbs_;
    double cpu_, ram_, disk_;
    std::vector<double> pd_, pe_, pw_, px_;
};

}  // namespace ds4split
