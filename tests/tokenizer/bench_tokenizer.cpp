// Tokenizer microbench and large-corpus agreement check (docs/gemma4-plan.md M1-11). CPU only.
//
//   bench_tokenizer --corpus <jsonl> [--model-dir DIR] [--repeats N] [--check] [--bench]
//
// The corpus is JSONL {"name", "text", "ids"} written by tools/tok_ref/bench_hf.py (benchmark corpus)
// or tools/tok_ref/check_hf_equiv.py --emit-jsonl (HF agreement audit corpus); `ids` are HF's raw
// `tokenizers` ids with special tokens not parsed. Timed (best of --repeats passes, single thread):
//   encode         Tokenizer::encode(text)
//   decode         Tokenizer::decode(ids)
//   stream_decode  StreamDecoder::push(id) one id at a time + flush
// MB/s counts UTF-8 bytes of the corpus text, Mtok/s counts ids; tools/tok_ref/bench_hf.py prints
// the same metrics for the HF Rust backend. --check additionally requires encode(text) == ids and
// decode(ids) == text for every document (and stream decode == text), exit 1 on any difference;
// the timed passes are skipped when --check is given without --bench.
//
// First measurement (2026-10-01, this machine, single thread, Release clang-cl; tokenizers 0.22.2,
// transformers 5.5.0; corpus = 498 real docs / 6.51 MB / 2.0 M tokens from bench_hf.py; best of 5):
//                   r4dx          HF tokenizers (Rust)             ratio
//   encode          12.5 MB/s     4.45 MB/s (single-thread)        2.8x   (HF encode_batch, all cores: 22.6 MB/s)
//                   3.84 Mtok/s   1.37 Mtok/s
//   decode          361 MB/s      10.0 MB/s                         36x
//   stream_decode   264 MB/s      2.11 MB/s (DecodeStream.step)    125x
//   tokenizer load  0.7 s
// Plan target was >= 0.8x HF (single-thread encode); gate set from this run: r4dx encode >= 2.0x HF
// single-thread (>= ~9 MB/s here) and decode / stream_decode >= 10x. Re-measure on the same corpus
// after any change to bpe_tokenizer.cpp; numbers are machine-dependent, compare ratios.
#include <algorithm>
#include "r4dx/models_root.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "tokenizer.h"

using r4dx::TokenId;
using r4dx::Tokenizer;

namespace {

struct Doc {
    std::string name;
    std::string text;
    std::vector<TokenId> ids;
};

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
double best_of(int repeats, F&& fn) {
    double best = 1e30;
    for (int i = 0; i < repeats; ++i) {
        const double t0 = now_s();
        fn();
        best = std::min(best, now_s() - t0);
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = r4dx::ModelsPath("Huihui-gemma-4-12B-it-abliterated-tok");
    std::string corpus_path = r4dx::ModelsPath("r4dx/tok_bench/corpus_gemma.jsonl");
    int repeats = 5;
    bool check = false, bench = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--corpus") && i + 1 < argc) corpus_path = argv[++i];
        else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--check")) { check = true; bench = false; }
        else if (!std::strcmp(argv[i], "--bench")) bench = true;
        else {
            std::fprintf(stderr, "usage: bench_tokenizer [--corpus JSONL] [--model-dir DIR] [--repeats N] [--check] [--bench]\n");
            return 2;
        }
    }

    std::ifstream tf(model_dir + "/tokenizer.json", std::ios::binary);
    std::ifstream cf(corpus_path, std::ios::binary);
    if (!tf || !cf) {
        std::fprintf(stderr, "bench_tokenizer: SKIPPED -- need %s/tokenizer.json and %s\n", model_dir.c_str(),
                     corpus_path.c_str());
        return 77;
    }

    Tokenizer::Options opts;
    opts.allow_unimplemented_normalizer = true;
    const double t_load0 = now_s();
    Tokenizer tok = Tokenizer::from_directory(model_dir, opts);
    const double t_load = now_s() - t_load0;

    std::vector<Doc> docs;
    size_t nbytes = 0, ntok = 0;
    for (std::string line; std::getline(cf, line);) {
        if (line.empty()) continue;
        const nlohmann::json j = nlohmann::json::parse(line);
        Doc d;
        d.name = j.at("name").get<std::string>();
        d.text = j.at("text").get<std::string>();
        for (const auto& v : j.at("ids")) d.ids.push_back(v.get<TokenId>());
        nbytes += d.text.size();
        ntok += d.ids.size();
        docs.push_back(std::move(d));
    }
    std::printf("tokenizer load: %.2f s; corpus: %zu docs, %.3f MB utf-8, %zu tokens\n", t_load, docs.size(),
                nbytes / 1e6, ntok);

    int bad = 0;
    if (check) {
        for (const Doc& d : docs) {
            const std::vector<TokenId> got = tok.encode(d.text, /*parse_special=*/false);
            if (got != d.ids) {
                size_t k = 0;
                while (k < got.size() && k < d.ids.size() && got[k] == d.ids[k]) ++k;
                std::fprintf(stderr, "ENCODE MISMATCH [%s]: %zu vs %zu ids, first diff at %zu (got %d, want %d)\n",
                             d.name.c_str(), got.size(), d.ids.size(), k, k < got.size() ? got[k] : -1,
                             k < d.ids.size() ? d.ids[k] : -1);
                ++bad;
                continue;
            }
            // HF's decoder maps every U+2581 back to ' ' (also one that was literally in the text),
            // so decode(encode(text)) == text only up to that replacement.
            std::string expect_text;
            for (size_t k = 0; k < d.text.size();) {
                if (d.text.compare(k, 3, "\xE2\x96\x81") == 0) { expect_text += ' '; k += 3; }
                else expect_text += d.text[k++];
            }
            if (tok.decode(d.ids, true) != expect_text) {
                std::fprintf(stderr, "DECODE MISMATCH [%s]\n", d.name.c_str());
                ++bad;
            }
            auto dec = tok.make_stream_decoder(true);
            std::string s;
            for (TokenId id : d.ids) s += dec.push(id);
            s += dec.flush();
            if (s != expect_text) {
                std::fprintf(stderr, "STREAM DECODE MISMATCH [%s]\n", d.name.c_str());
                ++bad;
            }
        }
        std::printf("check: %d mismatch(es) over %zu documents\n", bad, docs.size());
    }

    if (bench) {
        size_t sink = 0;
        auto report = [&](const char* label, double dt) {
            std::printf("%-14s %8.2f MB/s   %7.3f Mtok/s   (%.1f ms per pass)\n", label, nbytes / 1e6 / dt,
                        ntok / 1e6 / dt, dt * 1000);
        };
        report("encode", best_of(repeats, [&] {
                   for (const Doc& d : docs) sink += tok.encode(d.text, false).size();
               }));
        report("decode", best_of(repeats, [&] {
                   for (const Doc& d : docs) sink += tok.decode(d.ids, true).size();
               }));
        report("stream_decode", best_of(repeats, [&] {
                   for (const Doc& d : docs) {
                       auto dec = tok.make_stream_decoder(true);
                       for (TokenId id : d.ids) sink += dec.push(id).size();
                       sink += dec.flush().size();
                   }
               }));
        if (sink == 1) std::printf("\n");  // keep the optimizer honest
    }
    return bad == 0 ? 0 : 1;
}
