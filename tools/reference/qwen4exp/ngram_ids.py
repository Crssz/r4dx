"""Bit-exact numpy port of the qwen4_exp n-gram row-id computation (plan 2.3).

Mirrors `Qwen4ExpTextNGramEmbedding` in transformers' `modular_qwen4_exp.py` (lines 562-708): the
splitmix64 multipliers, the prime head vocabularies after `ngram_vocab_size_base - 1`, the head offsets,
the padded row count, the EOS-segmented right shift and the hash `(t0*m0 ^ t1*m1 [^ t2*m2]) mod prime_h
+ offset_h`. Pure numpy int64, no torch, so the C++ host side (M1-9) can be checked against it.

    h = NGramHash.from_config_file(r"E:\\models\\r4dx\\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ\\config.json")
    ids, ctx = h.ngram_ids(tokens_chunk_1)            # [T1, 16] global row ids, last 2 history tokens
    ids, ctx = h.ngram_ids(tokens_chunk_2, ctx)       # identical to one call over both chunks
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> np.ndarray:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return np.array(multipliers, dtype=np.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def text_config_of(config: dict) -> dict:
    """The text config of a qwen4_exp `config.json` (top level for text-only configs)."""
    return config.get("text_config", config)


class NGramHash:
    """Row-id hashing of one PLE n-gram embedding (one entry of `ple_layer_ids`)."""

    def __init__(self, *, vocab_size: int, ngram_size: int, heads_per_ngram: int, ngram_vocab_size_base: int,
                 vocab_divisor: int, seed: int, eos_token_id: int, embed_dim: int, ple_layer_index: int = 0):
        self.vocab_size = int(vocab_size)
        self.ngram_size = int(ngram_size)
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = int(heads_per_ngram)
        self.heads = self.context_len * self.heads_per_ngram
        self.ngram_vocab_size_base = int(ngram_vocab_size_base)
        self.vocab_divisor = int(vocab_divisor)
        self.seed = int(seed)
        self.eos_token_id = int(eos_token_id)
        self.embed_dim = int(embed_dim)
        self.head_dim = self.embed_dim // self.heads
        self.ple_layer_index = int(ple_layer_index)

        primes, offsets, total = [], [], 0
        for head_idx in range(self.heads):
            global_head_idx = self.ple_layer_index * self.heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            primes.append(size)
            offsets.append(total)
            total += size
        self.head_primes = np.array(primes, dtype=np.int64)
        self.head_offsets = np.array(offsets, dtype=np.int64)
        self.total_rows = total
        self.padded_rows = math.ceil(total / self.vocab_divisor) * self.vocab_divisor
        self.multipliers = _build_layer_multipliers(self.vocab_size, self.ngram_size, self.ple_layer_index, self.seed)

    @classmethod
    def from_config(cls, config: dict, ple_layer_index: int = 0) -> "NGramHash":
        """From a parsed `config.json` (full or text config). Defaults are `Qwen4ExpTextConfig`'s."""
        tc = text_config_of(config)
        eos = tc["eos_token_id"]
        eos = eos[0] if isinstance(eos, list) else eos
        return cls(
            vocab_size=tc["vocab_size"],
            ngram_size=tc.get("ngram_size", 3),
            heads_per_ngram=tc.get("heads_per_ngram", 8),
            ngram_vocab_size_base=tc.get("ngram_vocab_size_base", 20_000_000),
            vocab_divisor=tc.get("make_ngram_vocab_size_divisible_by", 128),
            seed=tc.get("seed", 1234),
            eos_token_id=eos,
            embed_dim=tc.get("ple_embed_dim") or tc["hidden_size"],
            ple_layer_index=ple_layer_index,
        )

    @classmethod
    def from_config_file(cls, path: str | Path, ple_layer_index: int = 0) -> "NGramHash":
        return cls.from_config(json.loads(Path(path).read_text(encoding="utf-8")), ple_layer_index)

    def _shift_right_ignore_eos(self, token_ids: np.ndarray, shift: int) -> np.ndarray:
        """`token_ids[p - shift]`, or EOS when that crosses the start of p's EOS-delimited segment."""
        if shift == 0:
            return token_ids
        positions = np.arange(token_ids.shape[0], dtype=np.int64)
        eos_positions = np.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions)
        previous_eos = np.concatenate([np.array([-1], dtype=np.int64), previous_eos_inclusive[:-1]])
        position_in_segment = positions - (previous_eos + 1)
        source_positions = positions - shift
        shifted = token_ids[np.maximum(source_positions, 0)]
        valid = (position_in_segment >= shift) & (source_positions >= 0)
        return np.where(valid, shifted, np.int64(self.eos_token_id))

    def initial_context(self) -> np.ndarray:
        return np.full(self.context_len, self.eos_token_id, dtype=np.int64)

    def ngram_ids(self, token_ids, previous_context=None) -> tuple[np.ndarray, np.ndarray]:
        """Global row ids [T, heads] (int64) of one sequence chunk, and the context for the next chunk.

        `previous_context` is the last `ngram_size - 1` token ids of the history (None: start of sequence,
        all EOS), exactly what HF keeps in `conv_states[2]` of the PLE layer's cache.
        """
        tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        context = self.initial_context() if previous_context is None else np.asarray(previous_context, dtype=np.int64).reshape(-1)
        if context.shape[0] != self.context_len:
            raise ValueError(f"previous_context must hold {self.context_len} ids, got {context.shape[0]}")
        history = np.concatenate([context, tokens])
        shifted = [self._shift_right_ignore_eos(history, shift) for shift in range(self.ngram_size)]

        blocks = []
        # int64 products wrap exactly like torch's (they cannot overflow for ids < vocab_size by construction)
        with np.errstate(over="ignore"):
            for ngram in range(2, self.ngram_size + 1):
                start = (ngram - 2) * self.heads_per_ngram
                end = start + self.heads_per_ngram
                mixed = shifted[0] * self.multipliers[0]
                for position in range(1, ngram):
                    mixed = np.bitwise_xor(mixed, shifted[position] * self.multipliers[position])
                ids = np.remainder(mixed[:, None], self.head_primes[None, start:end])
                blocks.append(ids + self.head_offsets[None, start:end])
        ids = np.concatenate(blocks, axis=1)[self.context_len:]
        return ids, history[-self.context_len:].copy()
