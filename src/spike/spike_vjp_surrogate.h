#pragma once

#include <deque>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace neurong::spike_vjp {

struct SpikeVjpSurrogateConfig {
    double width_mv = 2.0;
    double param0 = 0.0;
    double param1 = 0.0;
};

struct SpikeVjpSurrogateRunParam {
    int npre = 0;
    const int* pre_node_indices = nullptr;
    const double* threshold = nullptr;
    const double* pre_v_tape = nullptr;
    double* pending_pre_spike_adj = nullptr;
    double* carry_v = nullptr;
};

struct SpikeVjpSurrogateOps {
    const char* name = nullptr;
    const char* source_file = nullptr;
    const char* source_symbol = nullptr;
    void (*run_cpu)(const SpikeVjpSurrogateRunParam&, const SpikeVjpSurrogateConfig&) = nullptr;
    void (*launch_gpu)(const SpikeVjpSurrogateRunParam&, const SpikeVjpSurrogateConfig&) = nullptr;
};

class SpikeVjpSurrogateRegistry {
public:
    static SpikeVjpSurrogateRegistry& getInstance();

    void registerSurrogate(
        SpikeVjpSurrogateOps ops,
        std::initializer_list<const char*> aliases);

    const SpikeVjpSurrogateOps* find(const std::string& name) const;
    const SpikeVjpSurrogateOps& require(const std::string& name) const;
    std::vector<std::string> names() const;

private:
    std::deque<SpikeVjpSurrogateOps> ops_;
    std::unordered_map<std::string, const SpikeVjpSurrogateOps*> alias2ops_;
};

std::string normalize_spike_vjp_surrogate_name(std::string name);

}  // namespace neurong::spike_vjp
