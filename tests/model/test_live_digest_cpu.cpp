// tests/model/test_live_digest_cpu.cpp -- CPU-only checks of src/model/live_digest.h and live_state.h, the host side of the hybrid
// mode's byte-identity gate G-H1 (docs/pp-tp2-hybrid.md 4):
//   * the canonical digest sees exactly the LIVE state: every live KV byte moves it, no byte of a dead row (past the position, in
//     the last block or beyond) does; head order, row order and layer matter; the MTP head's one unprimed row is dead;
//   * a device-shaped holder (KV cache bigger than the sequence with garbage past it, a recurrent buffer with stale window slots,
//     conv lines longer than the three live entries at the offset a speculative round left them) digested the way
//     Model::DebugLiveStateDigest does equals the digest of the host image of the same state, for a full holder and for each rank;
//   * the DFlash window rule (a tail-fed drafter and a fully fed one see the same window);
//   * the dump format: round trip of an image with scalars and extra payloads, damage / truncation / duplicates / unknown records are
//     refused; the canonical form is idempotent;
//   * NEGATIVE CONTROLS: a wrong conv offset, a stale window slot, swapped heads, a live byte flipped each break equality.
// No HIP call, no container; always runs.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "live_state.h"

using namespace r4dx::model;
using namespace r4dx::model::hybrid;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}
template <class F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

std::mt19937_64 g_rng(20261008);
std::vector<uint8_t> RandBytes(size_t n) {
  std::vector<uint8_t> v(n);
  for (uint8_t& b : v) b = static_cast<uint8_t>(g_rng());
  return v;
}
std::vector<uint16_t> RandU16(size_t n) {
  std::vector<uint16_t> v(n);
  for (uint16_t& b : v) b = static_cast<uint16_t>(g_rng());
  return v;
}

// The tiny model of test_reshard_plan_cpu.cpp: 8 layers (attention at 3 and 7), 4 kv heads of head_dim 4, 4 key / 8 value heads of 4.
ModelConfig TinyConfig() {
  ModelConfig c;
  c.hidden_size = 32;
  c.num_hidden_layers = 8;
  for (int i = 0; i < 8; ++i) c.layer_types.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  c.num_attention_heads = 8;
  c.num_key_value_heads = 4;
  c.head_dim = 4;
  c.intermediate_size = 16;
  c.linear_num_key_heads = 4;
  c.linear_num_value_heads = 8;
  c.linear_key_head_dim = 4;
  c.linear_value_head_dim = 4;
  c.vocab_size = 64;
  return c;
}

// ---- 1. the KV digest sees exactly the live rows ----------------------------------------------------------------------------
void KvLiveBytesOnly() {
  const int64_t H = 3, bt = 4, rb = 6, rows = 10;  // 3 blocks, the last holds 2 live rows
  const std::vector<uint8_t> base = RandBytes(static_cast<size_t>(3 * H * bt * rb));
  const uint64_t d0 = DigestKvImage(base, H, bt, rb, rows);
  size_t live = 0, dead = 0;
  bool live_moves = true, dead_still = true;
  for (size_t i = 0; i < base.size(); ++i) {
    const int64_t block = static_cast<int64_t>(i) / (H * bt * rb);
    const int64_t row = block * bt + (static_cast<int64_t>(i) / rb) % bt;
    std::vector<uint8_t> x = base;
    x[i] ^= 0x5A;
    const bool moves = DigestKvImage(x, H, bt, rb, rows) != d0;
    if (row < rows) {
      ++live;
      live_moves = live_moves && moves;
    } else {
      ++dead;
      dead_still = dead_still && !moves;
    }
  }
  Check(live == static_cast<size_t>(H * rows * rb) && dead == static_cast<size_t>(H * (3 * bt - rows) * rb), "KV digest: the live / dead byte counts are H * rows * row_bytes");
  Check(live_moves, "KV digest: flipping ANY live byte (every head, every row < pos) changes it");
  Check(dead_still, "KV digest: flipping ANY dead byte (rows >= pos, in the last block) leaves it unchanged");
  // more blocks than needed (a bigger cache): the extra blocks are ignored
  std::vector<uint8_t> big = base;
  big.resize(base.size() + static_cast<size_t>(5 * H * bt * rb), 0xCD);
  Check(DigestKvImage(big, H, bt, rb, rows) == d0, "KV digest: blocks past the last live one are ignored");
  // rows == 0 and a short image
  Check(DigestKvImage({}, H, bt, rb, 0) == DigestKvImage(RandBytes(static_cast<size_t>(H * bt * rb)), H, bt, rb, 0), "KV digest: zero rows hash equal whatever the image holds");
  Check(Throws([&] { (void)DigestKvImage(std::vector<uint8_t>(static_cast<size_t>(H * bt * rb)), H, bt, rb, rows); }), "KV digest: an image with too few blocks is refused");
  // head order and a different row count
  std::vector<uint8_t> swapped = base;
  for (int64_t b = 0; b < 3; ++b) {
    for (int64_t k = 0; k < bt * rb; ++k) std::swap(swapped[static_cast<size_t>((b * H + 0) * bt * rb + k)], swapped[static_cast<size_t>((b * H + 1) * bt * rb + k)]);
  }
  Check(DigestKvImage(swapped, H, bt, rb, rows) != d0, "NEGATIVE CONTROL: swapping two heads changes the digest (head order is part of the record order)");
  Check(DigestKvImage(base, H, bt, rb, rows - 1) != d0 && DigestKvImage(base, H, bt, rb, rows + 1) != d0, "KV digest: the row count is part of it");

  // incremental feeding in any grouping gives the same value
  bool groups_equal = true;
  for (int trial = 0; trial < 60; ++trial) {
    KvDigester d(H, bt, rb, rows);
    int64_t b = 0;
    while (b < d.BlocksNeeded()) {
      const int64_t n = 1 + static_cast<int64_t>(g_rng() % 4);
      d.Feed(base.data() + static_cast<size_t>(b * H * bt * rb), b, n);
      b = d.NextBlock();
    }
    groups_equal = groups_equal && d.Finish() == d0;
  }
  Check(groups_equal, "KvDigester: any grouping of the blocks gives the one-shot digest");
  KvDigester gap(H, bt, rb, rows);
  Check(Throws([&] { gap.Feed(base.data(), 1, 1); }), "KvDigester: a gap in the block sequence is refused");
  KvDigester early(H, bt, rb, rows);
  early.Feed(base.data(), 0, 1);
  Check(Throws([&] { (void)early.Finish(); }), "KvDigester: Finish before every needed block was fed is refused");
}

// ---- 2. a device-shaped holder digests like its host image ---------------------------------------------------------------------
// The holder as Model::DebugLiveStateDigest sees it: KV caches of `max_blocks` with garbage past the live rows, a recurrent buffer of
// `slots` slots (the live one at `live_slot`, stale garbage elsewhere), conv lines of `pitch` entries per channel whose live entries
// start at `conv_off` (the rest garbage).
struct DeviceHolder {
  std::map<int64_t, std::vector<uint8_t>> kv;
  std::vector<uint8_t> mtp;
  std::map<int64_t, std::vector<uint8_t>> rec;   // slots x slot bytes
  std::map<int64_t, std::vector<uint16_t>> conv; // channels x pitch
  int64_t slots = 3, live_slot = 1, pitch = 6, conv_off = 2;
};

DeviceHolder MakeDevice(const LiveImage& img, const DigestShape& sh, int64_t pos, int64_t max_blocks, int64_t slots, int64_t live_slot, int64_t pitch, int64_t conv_off) {
  DeviceHolder d;
  d.slots = slots;
  d.live_slot = live_slot;
  d.pitch = pitch;
  d.conv_off = conv_off;
  const size_t block = static_cast<size_t>(sh.kv_heads * sh.block_tokens * sh.kv_row_bytes);
  const auto cache = [&](const std::vector<uint8_t>& image) {
    std::vector<uint8_t> c = RandBytes(static_cast<size_t>(max_blocks) * block);  // garbage everywhere ...
    std::memcpy(c.data(), image.data(), image.size());                             // ... then the live blocks (their dead tail rows stay canonical zero)
    return c;
  };
  for (const auto& [l, b] : img.kv) d.kv[l] = cache(b);
  if (!img.mtp_kv.empty()) d.mtp = cache(img.mtp_kv);
  for (const auto& [l, b] : img.gdn_rec) {
    std::vector<uint8_t> r = RandBytes(static_cast<size_t>(slots) * b.size());
    std::memcpy(r.data() + static_cast<size_t>(live_slot) * b.size(), b.data(), b.size());
    d.rec[l] = std::move(r);
  }
  for (const auto& [l, c] : img.gdn_conv) {
    const int64_t channels = static_cast<int64_t>(c.size()) / sh.conv_live;
    std::vector<uint16_t> line = RandU16(static_cast<size_t>(channels * pitch));
    for (int64_t ch = 0; ch < channels; ++ch) {
      for (int64_t e = 0; e < sh.conv_live; ++e) line[static_cast<size_t>(ch * pitch + conv_off + e)] = c[static_cast<size_t>(ch * sh.conv_live + e)];
    }
    d.conv[l] = std::move(line);
  }
  return d;
}

// What Model::DebugLiveStateDigest computes, over host copies of the holder's buffers.
DigestList DigestDevice(const DeviceHolder& d, const DigestShape& sh, int64_t pos, int64_t group, size_t slot_bytes, int64_t read_slot, int64_t read_off) {
  const auto kv_digest = [&](const std::vector<uint8_t>& cache, int64_t rows) {
    KvDigester dg(sh.kv_heads, sh.block_tokens, sh.kv_row_bytes, rows);
    const int64_t block = sh.kv_heads * sh.block_tokens * sh.kv_row_bytes;
    for (int64_t b = 0; b < dg.BlocksNeeded(); b += group) {
      const int64_t n = std::min(group, dg.BlocksNeeded() - b);
      dg.Feed(cache.data() + static_cast<size_t>(b * block), b, n);
    }
    return dg.Finish();
  };
  DigestList out;
  out.emplace_back("pos", static_cast<uint64_t>(pos));
  for (const auto& [l, c] : d.kv) out.emplace_back(DigestName("kv", l), kv_digest(c, pos));
  if (!d.mtp.empty() && MtpLiveRows(pos) > 0) out.emplace_back("mtp.kv", kv_digest(d.mtp, MtpLiveRows(pos)));
  for (const auto& [l, r] : d.rec) out.emplace_back(DigestName("gdn.rec", l), DigestBytes(r.data() + static_cast<size_t>(read_slot) * slot_bytes, slot_bytes));
  for (const auto& [l, line] : d.conv) {
    const int64_t channels = static_cast<int64_t>(line.size()) / d.pitch;
    std::vector<uint16_t> compact(static_cast<size_t>(channels * sh.conv_live));
    for (int64_t ch = 0; ch < channels; ++ch) {
      for (int64_t e = 0; e < sh.conv_live; ++e) compact[static_cast<size_t>(ch * sh.conv_live + e)] = line[static_cast<size_t>(ch * d.pitch + read_off + e)];
    }
    out.emplace_back(DigestName("gdn.conv", l), DigestBytes(compact.data(), compact.size() * sizeof(uint16_t)));
  }
  return out;
}

LiveImage TruthImage(const StateGeometry& g, int64_t pos) {
  LiveImage x;
  const size_t row_bytes = static_cast<size_t>(g.KvTokenHeadBytes());
  const auto kv = [&](int64_t rows) {
    std::vector<uint8_t> v = RandBytes(static_cast<size_t>(((rows + g.block_tokens - 1) / g.block_tokens) * g.KvBlockBytesFull()));
    CanonicalizeKv(&v, g.kv_heads_full, g.block_tokens, static_cast<int64_t>(row_bytes), rows);
    return v;
  };
  for (const int64_t l : g.attn_layers) x.kv[l] = kv(pos);
  x.mtp_kv = kv(MtpLiveRows(pos));
  for (const int64_t l : g.gdn_layers) {
    x.gdn_rec[l] = RandBytes(static_cast<size_t>(g.RecurrentBytesFull()));
    x.gdn_conv[l] = RandU16(static_cast<size_t>(g.conv_dim_full * g.conv_live));
  }
  return x;
}

void DeviceMatchesImage() {
  const StateGeometry g = StateGeometry::FromRules(TinyConfig(), /*block_tokens=*/4);
  for (const int64_t pos : {1, 2, 5, 9, 16, 23}) {
    const LiveImage full = TruthImage(g, pos);
    for (int holder = -1; holder < 2; ++holder) {  // -1: the full-head holder (a TP=1 Model / a stage), 0 / 1: the ranks
      const LiveImage img = holder < 0 ? full : ReshardRef(g, full, holder);
      DigestShape sh;
      sh.block_tokens = g.block_tokens;
      sh.kv_heads = holder < 0 ? g.kv_heads_full : g.ranks[static_cast<size_t>(holder)].kv_heads;
      sh.kv_row_bytes = g.KvTokenHeadBytes();
      sh.conv_live = g.conv_live;
      const size_t slot_bytes = img.gdn_rec.begin()->second.size();
      const DigestList want = DigestLiveImage(img, sh, pos);
      // plain: window 0, offset 0, compact lines; speculative: window slot 2, conv offset 2 of 6-entry lines
      struct Case {
        int64_t slots, live_slot, pitch, off, group;
      };
      for (const Case c : {Case{1, 0, 3, 0, 1}, Case{4, 0, 6, 0, 3}, Case{4, 2, 6, 2, 2}, Case{4, 3, 5, 1, 7}}) {
        const DeviceHolder dev = MakeDevice(img, sh, pos, /*max_blocks=*/9, c.slots, c.live_slot, c.pitch, c.off);
        const DigestList got = DigestDevice(dev, sh, pos, c.group, slot_bytes, c.live_slot, c.off);
        const std::string why = DiffDigests(want, got);
        if (!why.empty()) std::fprintf(stderr, "  holder %d pos %lld slots %lld: %s\n", holder, static_cast<long long>(pos), static_cast<long long>(c.slots), why.c_str());
        Check(why.empty(), "a device-shaped holder (garbage past the rows, stale slots, long conv lines) digests exactly like its host image");
        if (c.slots > 1) {
          // negative controls: the digest of the wrong window slot / conv offset differs
          const DigestList stale_slot = DigestDevice(dev, sh, pos, c.group, slot_bytes, (c.live_slot + 1) % c.slots, c.off);
          Check(!DiffDigests(want, stale_slot).empty(), "NEGATIVE CONTROL: reading a stale window slot changes the recurrent digest");
          const DigestList wrong_off = DigestDevice(dev, sh, pos, c.group, slot_bytes, c.live_slot, (c.off + 1) % (c.pitch - sh.conv_live + 1));
          if (c.pitch > sh.conv_live) Check(!DiffDigests(want, wrong_off).empty(), "NEGATIVE CONTROL: reading the conv history at the wrong offset changes the conv digest");
        }
      }
      // a flipped live byte in the image moves its record; a flipped dead KV byte does not
      LiveImage bad = img;
      bad.gdn_rec.begin()->second[0] ^= 1;
      Check(!DiffDigests(want, DigestLiveImage(bad, sh, pos)).empty(), "NEGATIVE CONTROL: a flipped recurrent byte changes the digest");
      LiveImage dead = img;
      const int64_t rem = pos % g.block_tokens;
      if (rem != 0) {
        std::vector<uint8_t>& k = dead.kv.begin()->second;
        const size_t head_block = static_cast<size_t>(g.block_tokens * g.KvTokenHeadBytes());
        const size_t last_block = k.size() - static_cast<size_t>(sh.kv_heads) * head_block;
        k[last_block + static_cast<size_t>(rem * g.KvTokenHeadBytes())] ^= 0xFF;  // head 0, first dead row of the last block
        Check(DiffDigests(want, DigestLiveImage(dead, sh, pos)).empty(), "a flipped DEAD KV byte leaves the digest unchanged");
      }
    }
  }
  // the TP=1 reference and the ranks: the rank digests are computed on the rank images of ONE truth (the gate's shape)
  const LiveImage truth = TruthImage(g, 13);
  DigestShape full_sh{g.block_tokens, g.kv_heads_full, g.KvTokenHeadBytes(), g.conv_live};
  DigestShape rank_sh = full_sh;
  rank_sh.kv_heads = g.ranks[0].kv_heads;
  Check(DiffDigests(DigestLiveImage(ReshardRef(g, truth, 0), rank_sh, 13), DigestLiveImage(ReshardRef(g, truth, 1), rank_sh, 13)) != "", "the two ranks' digests differ (different halves)");
  LiveImage broken = truth;
  std::swap(broken.kv.begin()->second, broken.kv.rbegin()->second);
  Check(DiffDigests(DigestLiveImage(ReshardRef(g, truth, 0), rank_sh, 13), DigestLiveImage(ReshardRef(g, broken, 0), rank_sh, 13)) != "", "NEGATIVE CONTROL: two layers' KV swapped changes a rank digest (the layer is part of the record)");
}

// ---- 3. MTP live rows, the DFlash window ------------------------------------------------------------------------------------------
void MtpAndDflash() {
  Check(MtpLiveRows(0) == 0 && MtpLiveRows(1) == 0 && MtpLiveRows(2) == 1 && MtpLiveRows(100) == 99, "the MTP head's live rows are pos - 1 (row pos - 1 is primed by the next call)");
  int64_t lo = 0, hi = 0;
  DflashWindow(0, 0, 2048, &lo, &hi);
  Check(lo == 0 && hi == 0, "dflash window: nothing injected is empty");
  DflashWindow(1500, 0, 2048, &lo, &hi);
  Check(lo == 0 && hi == 1500, "dflash window: fewer rows than slots is everything");
  DflashWindow(5000, 0, 2048, &lo, &hi);
  Check(lo == 2952 && hi == 5000, "dflash window: the last 2048");
  int64_t lo2 = 0, hi2 = 0;
  DflashWindow(5000, 2900, 2048, &lo2, &hi2);  // a tail-only drafter whose gap opened at 2900 <= 5000 - 2048
  Check(lo2 == lo && hi2 == hi, "dflash window: a drafter fed only a tail of >= 2048 rows sees the same window as one fed everything");
  DflashWindow(5000, 3100, 2048, &lo2, &hi2);
  Check(lo2 == 3100, "dflash window: a gap inside the last 2048 shortens the visible store (and shows in the digest)");
  const std::vector<uint16_t> rows = RandU16(40);
  Check(DigestDflashRows(rows.data(), 5, 8, 100) != DigestDflashRows(rows.data(), 5, 8, 101), "dflash rows digest: the first position is part of it");
  Check(DigestDflashRows(rows.data(), 5, 8, 100) == DigestDflashRows(rows.data(), 5, 8, 100), "dflash rows digest: deterministic");
}

// ---- 4. canonical form and the dump file --------------------------------------------------------------------------------------------
void Canonical() {
  const int64_t H = 2, bt = 4, rb = 3;
  std::vector<uint8_t> v = RandBytes(static_cast<size_t>(4 * H * bt * rb));
  CanonicalizeKv(&v, H, bt, rb, 9);  // 9 rows -> 3 blocks, 1 live row in the last
  Check(v.size() == static_cast<size_t>(3 * H * bt * rb), "canonical KV: blocks past the last live one are dropped");
  bool zeros = true, live_kept = true;
  const std::vector<uint8_t> again = [&] {
    std::vector<uint8_t> c = v;
    CanonicalizeKv(&c, H, bt, rb, 9);
    return c;
  }();
  Check(again == v, "canonical KV: idempotent");
  for (int64_t h = 0; h < H; ++h) {
    for (int64_t t = 0; t < bt; ++t) {
      for (int64_t k = 0; k < rb; ++k) {
        const uint8_t b = v[static_cast<size_t>(((2 * H + h) * bt + t) * rb + k)];
        if (t >= 1) zeros = zeros && b == 0;
      }
    }
  }
  Check(zeros, "canonical KV: the dead rows of the last block are zero");
  (void)live_kept;
  std::vector<uint8_t> exact = RandBytes(static_cast<size_t>(2 * H * bt * rb));
  const std::vector<uint8_t> copy = exact;
  CanonicalizeKv(&exact, H, bt, rb, 8);
  Check(exact == copy, "canonical KV: an exactly full last block is untouched");
  Check(Throws([&] {
    std::vector<uint8_t> small(static_cast<size_t>(H * bt * rb));
    CanonicalizeKv(&small, H, bt, rb, 9);
  }), "canonical KV: an image with too few blocks is refused");
}

LiveState MakeState() {
  const StateGeometry g = StateGeometry::FromRules(TinyConfig(), 4);
  LiveState st;
  st.image = TruthImage(g, 11);
  st.scalars.pos = 11;
  st.scalars.started = true;
  st.scalars.mrope_active = true;
  st.scalars.mrope_delta = -37;
  st.scalars.mtp_seed_valid = true;
  st.scalars.mtp_seed = RandU16(32);
  st.extra["features"] = RandBytes(500);
  st.extra["rope_t"] = RandBytes(44);
  return st;
}
bool SameState(const LiveState& a, const LiveState& b) {
  return a.image == b.image && a.scalars.pos == b.scalars.pos && a.scalars.started == b.scalars.started && a.scalars.mrope_active == b.scalars.mrope_active &&
         a.scalars.mrope_delta == b.scalars.mrope_delta && a.scalars.mtp_seed_valid == b.scalars.mtp_seed_valid && a.scalars.mtp_seed == b.scalars.mtp_seed &&
         a.extra == b.extra;
}

void DumpFile() {
  const LiveState st = MakeState();
  const std::vector<uint8_t> bytes = EncodeBlobs(LiveStateToBlobs(st));
  Check(SameState(LiveStateFromBlobs(DecodeBlobs(bytes.data(), bytes.size())), st), "dump: the state (image, scalars with the seed, extra payloads) round-trips through the blob container");
  // damage
  bool all_refused = true;
  for (size_t i = 0; i < bytes.size(); i += std::max<size_t>(1, bytes.size() / 97)) {
    std::vector<uint8_t> x = bytes;
    x[i] ^= 0x40;
    all_refused = all_refused && Throws([&] { (void)DecodeBlobs(x.data(), x.size()); });
  }
  Check(all_refused, "dump: a flipped byte anywhere is caught (magic or checksum)");
  Check(Throws([&] { (void)DecodeBlobs(bytes.data(), bytes.size() - 1); }), "dump: a truncated file is refused");
  Check(Throws([&] { (void)DecodeBlobs(bytes.data(), 5); }), "dump: a file shorter than its header is refused");
  // duplicates and unknown records (with a VALID checksum: they are structural errors)
  std::vector<Blob> dup = LiveStateToBlobs(st);
  dup.push_back(dup.back());
  const std::vector<uint8_t> dup_bytes = EncodeBlobs(dup);
  Check(Throws([&] { (void)DecodeBlobs(dup_bytes.data(), dup_bytes.size()); }), "dump: a duplicate record name is refused");
  std::vector<Blob> unk = LiveStateToBlobs(st);
  unk.push_back({"surprise", {1, 2, 3}});
  Check(Throws([&] { (void)LiveStateFromBlobs(unk); }), "dump: an unknown record is refused");
  std::vector<Blob> no_scalars = LiveStateToBlobs(st);
  no_scalars.erase(no_scalars.begin());
  Check(Throws([&] { (void)LiveStateFromBlobs(no_scalars); }), "dump: a file without scalars is refused");
  // through a real file
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "r4dx_test_live_digest_cpu.state";
  WriteLiveState(path.string(), st);
  Check(SameState(ReadLiveState(path.string()), st), "dump: write + read through a file");
  std::filesystem::remove(path);
  Check(Throws([&] { (void)ReadLiveState(path.string()); }), "dump: reading a missing file is an error");
  // a seedless state
  LiveState bare = st;
  bare.scalars.mtp_seed_valid = false;
  bare.scalars.mtp_seed.clear();
  bare.extra.clear();
  const std::vector<uint8_t> bare_bytes = EncodeBlobs(LiveStateToBlobs(bare));
  Check(SameState(LiveStateFromBlobs(DecodeBlobs(bare_bytes.data(), bare_bytes.size())), bare), "dump: a state without a seed or extras round-trips");
}

void Diff() {
  const DigestList a = {{"pos", 5}, {"kv.3", 7}};
  Check(DiffDigests(a, {{"kv.3", 7}, {"pos", 5}}).empty(), "DiffDigests: order is irrelevant");
  Check(!DiffDigests(a, {{"pos", 5}, {"kv.3", 8}}).empty(), "DiffDigests: a value difference is reported");
  Check(!DiffDigests(a, {{"pos", 5}}).empty() && !DiffDigests({{"pos", 5}}, a).empty(), "DiffDigests: a missing record is reported from both sides");
}

}  // namespace

int main() {
  KvLiveBytesOnly();
  DeviceMatchesImage();
  MtpAndDflash();
  Canonical();
  DumpFile();
  Diff();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_live_digest_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_live_digest_cpu: PASS\n");
  return 0;
}
