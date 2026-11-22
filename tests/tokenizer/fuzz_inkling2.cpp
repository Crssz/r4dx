// tokenizer_fuzz_inkling2 <corpus.jsonl> <model-dir>: replays a JSONL corpus written by tools/tok_ref/fuzz_inkling2.py
// ({"text": ..., "parse_special": bool, "ids": [...HF ids...]}) through r4dx's Tokenizer and reports every id mismatch.
// Not a CTest test (the corpus is generated on demand against the real HF tokenizer); exits 1 on any mismatch.
#include <cstdio>
#include <fstream>
#include <string>

#include "nlohmann/json.hpp"
#include "tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: tokenizer_fuzz_inkling2 <corpus.jsonl> <model-dir>\n");
        return 2;
    }
    r4dx::Tokenizer tok;
    try {
        tok = r4dx::Tokenizer::from_directory(argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load failed: %s\n", e.what());
        return 1;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    std::string line;
    size_t n = 0, bad = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const nlohmann::json j = nlohmann::json::parse(line);
        const std::string text = j.at("text").get<std::string>();
        const std::vector<r4dx::TokenId> expected = j.at("ids").get<std::vector<r4dx::TokenId>>();
        const std::vector<r4dx::TokenId> got = tok.encode(text, j.at("parse_special").get<bool>());
        ++n;
        if (got != expected) {
            if (++bad <= 20) {
                std::fprintf(stderr, "MISMATCH case %zu: text=%s\n  expected:", n - 1, nlohmann::json(text).dump().c_str());
                for (const auto id : expected) std::fprintf(stderr, " %d", id);
                std::fprintf(stderr, "\n  got     :");
                for (const auto id : got) std::fprintf(stderr, " %d", id);
                std::fprintf(stderr, "\n");
            }
        }
    }
    std::printf("tokenizer_fuzz_inkling2: %zu cases, %zu mismatches\n", n, bad);
    return bad == 0 ? 0 : 1;
}
