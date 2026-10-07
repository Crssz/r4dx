// r4dx::kernels argmax references -- host-only (no HIP), shared by tests/kernels/test_argmax_order_cpu.cpp
// (the order algebra) and tests/kernels/test_argmax_multi.cpp (the device kernels against it).
//
// What r4dx_argmax_f32 / r4dx_argmax_val_f32 compute, for every input, is
//   (value, index) = the maximum, under the total order "larger value first, then lower index", of
//                    the elements that compare greater than the seed (-inf, 0), and the seed itself
//                    when there are none (so a NaN never wins, -inf is never a winner on its own, and a
//                    row of -inf / NaN gives (-inf, 0)).
// Both the one-workgroup kernel (ArgmaxKernel: per-thread strided scan, warp xor tree, block tree) and
// the multi-workgroup pair (ArgmaxPartialKernel / ArgmaxFinalKernel) only ever apply ArgmaxTakeRef to
// pairs, so each is that maximum for any partition of the elements and any combine order -- which is
// what DirectRef states and PartitionedRef exercises.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace r4dx::kernels {

struct ArgmaxPairRef {
  float v = -std::numeric_limits<float>::infinity();
  int32_t i = 0;
};

// One step of every reduction in the device kernels (ArgmaxTake): a larger value wins, an equal value
// the lower index; NaN compares false both ways, so it never replaces anything.
inline void ArgmaxTakeRef(ArgmaxPairRef& best, float v, int32_t i) {
  if (v > best.v || (v == best.v && i < best.i)) {
    best.v = v;
    best.i = i;
  }
}

// The definition, element by element in index order.
inline ArgmaxPairRef ArgmaxDirectRef(const float* x, int64_t n) {
  ArgmaxPairRef best;
  for (int64_t i = 0; i < n; ++i) ArgmaxTakeRef(best, x[i], static_cast<int32_t>(i));
  return best;
}

// The one-workgroup kernel's own order: `threads` strided scanners (ascending within a thread), then
// the pairwise xor-tree over the per-thread partials (the tree's shape is irrelevant to the result, but
// is reproduced: lane l combines with lane l ^ o for o = 16..1 within a 32-lane group, then the group
// results combine the same way).
inline ArgmaxPairRef ArgmaxSingleWgRef(const float* x, int64_t n, int threads = 256) {
  std::vector<ArgmaxPairRef> p(static_cast<size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    for (int64_t i = t; i < n; i += threads) ArgmaxTakeRef(p[static_cast<size_t>(t)], x[i], static_cast<int32_t>(i));
  }
  const auto tree = [](std::vector<ArgmaxPairRef>& v) {  // v.size() a multiple of 32 or exactly 32
    const size_t lanes = v.size();
    for (size_t base = 0; base < lanes; base += 32) {
      for (int o = 16; o; o >>= 1) {
        std::vector<ArgmaxPairRef> next(v.begin() + static_cast<long>(base), v.begin() + static_cast<long>(base) + 32);
        for (int l = 0; l < 32; ++l) {
          ArgmaxPairRef b = v[base + static_cast<size_t>(l)];
          const ArgmaxPairRef& other = v[base + static_cast<size_t>(l ^ o)];
          ArgmaxTakeRef(b, other.v, other.i);
          next[static_cast<size_t>(l)] = b;
        }
        for (int l = 0; l < 32; ++l) v[base + static_cast<size_t>(l)] = next[static_cast<size_t>(l)];
      }
    }
  };
  tree(p);
  std::vector<ArgmaxPairRef> w(32);
  for (size_t g = 0; g < static_cast<size_t>(threads) / 32 && g < 32; ++g) w[g] = p[g * 32];
  tree(w);
  return w[0];
}

// Any partition of [0, n) into contiguous parts at `cuts` (ascending interior boundaries), partial per
// part, partials combined in `order` (a permutation of the part indices): the multi-workgroup shape.
inline ArgmaxPairRef ArgmaxPartitionedRef(const float* x, int64_t n, const std::vector<int64_t>& cuts,
                                           const std::vector<size_t>& order) {
  std::vector<int64_t> b;
  b.push_back(0);
  for (int64_t c : cuts) b.push_back(c);
  b.push_back(n);
  std::vector<ArgmaxPairRef> part(b.size() - 1);
  for (size_t k = 0; k + 1 < b.size(); ++k) {
    for (int64_t i = b[k]; i < b[k + 1]; ++i) ArgmaxTakeRef(part[k], x[i], static_cast<int32_t>(i));
  }
  ArgmaxPairRef best;
  for (size_t k : order) ArgmaxTakeRef(best, part[k].v, part[k].i);
  return best;
}

}  // namespace r4dx::kernels
