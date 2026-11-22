#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
int main(int argc, char ** argv) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    if (argc > 3) mp.lazy_mode = (llama_lazy_mode) atoi(argv[3]);
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    if (!m) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = atoi(argv[2]);
    llama_context * c = llama_init_from_model(m, cp);
    const llama_vocab * v = llama_model_get_vocab(m);
    const int nv = llama_vocab_n_tokens(v);
    std::mt19937 gen(7);
    std::vector<llama_token> toks(150);
    for (auto & t : toks) t = gen() % nv;
    FILE * f = fopen(argv[4], "wb");
    for (size_t i = 0; i < toks.size(); i += 50) {
        llama_batch b = llama_batch_get_one(toks.data() + i, 50);
        if (llama_decode(c, b)) return 2;
        const float * l = llama_get_logits_ith(c, -1);
        fwrite(l, sizeof(float), nv, f);
        printf("%zu: %.6f %.6f %.6f\n", i, l[0], l[1], l[nv-1]);
    }
    fclose(f);
    llama_free(c); llama_model_free(m);
    return 0;
}
