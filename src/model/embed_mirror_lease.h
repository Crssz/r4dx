// The device mirror of text.embed_tokens as something a second Container on the SAME device can borrow
// (docs/pp-tp2-hybrid.md 7 "Destruction and borrowed state", P0 item 12): in the hybrid mode each card holds a TP rank Model
// and a stage Model, and the stage borrows the rank Container's 2.37 GiB mirror instead of uploading its own.
//
// The owner (the rank's Container) creates one lease when it uploads the mirror and hands out copies of the shared_ptr
// (Container::EmbedMirrorLeaseHandle); a borrower keeps its copy for as long as it reads the pointer. The owner's destructor
// asserts that no copy is left (EmbedMirrorLeaseViolation): freeing the mirror under a live borrower would leave the stage
// gathering from freed device memory, so the rule is "the stage Model is destroyed before the rank Model that owns the
// mirror" and a violation aborts with the message instead of corrupting a later request. Header-only and HIP-free (the device
// is just an int here) so tests/model/test_stage_load_cpu.cpp covers the counting.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace r4dx::model {

struct EmbedMirrorLease {
  const uint16_t* data = nullptr;  // the mirror, bf16 [vocab, hidden], on `device`
  size_t elems = 0;
  int device = -1;                 // the HIP device that owns it: a borrower on another device would read foreign memory
};

// Copies of the lease alive besides the owner's own.
inline long EmbedMirrorBorrowers(const std::shared_ptr<EmbedMirrorLease>& owners_handle) {
  return owners_handle ? owners_handle.use_count() - 1 : 0;
}

// "" when nothing borrows the mirror any more (or there is no lease); else the message the owner's destructor aborts with.
inline std::string EmbedMirrorLeaseViolation(const std::shared_ptr<EmbedMirrorLease>& owners_handle) {
  const long n = EmbedMirrorBorrowers(owners_handle);
  if (n <= 0) return "";
  return "the embedding device mirror is being destroyed while " + std::to_string(n) +
         " other Container(s) still borrow it: destroy the stage Model before the rank Model that owns the mirror";
}

// "" when a Container on `device` may borrow `lease` for a text.embed_tokens of `elems` elements.
inline std::string CheckEmbedMirrorBorrow(const EmbedMirrorLease& lease, int device, size_t elems) {
  if (lease.data == nullptr || lease.elems == 0) return "the borrowed embedding mirror is empty";
  if (lease.device != device) {
    return "the embedding mirror lives on HIP device " + std::to_string(lease.device) + ", this Container loads on device " +
           std::to_string(device);
  }
  if (lease.elems != elems) {
    return "the embedding mirror holds " + std::to_string(lease.elems) + " elements, text.embed_tokens has " + std::to_string(elems);
  }
  return "";
}

}  // namespace r4dx::model
