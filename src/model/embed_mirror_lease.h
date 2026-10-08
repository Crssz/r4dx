// The device mirror of text.embed_tokens as something a second Container on the SAME device can borrow
// (docs/pp-tp2-hybrid.md 7 "Destruction and borrowed state", P0 item 12): in the hybrid mode each card holds a TP rank Model
// and a stage Model, and the stage borrows the rank Container's 2.37 GiB mirror instead of uploading its own.
//
// The lease OWNS the mirror (`keepalive` is the shared owner of the DeviceBuffer): the uploading Container and every borrower
// hold a copy of the shared_ptr, and the device memory is freed when the last copy goes. No destruction order is required --
// a stage Model that outlives the rank Model that uploaded the mirror (or a ModelOptions the caller kept) still gathers from
// live memory; the VRAM is simply returned later. Header-only and HIP-free (the device is just an int, the buffer a type-erased
// owner) so tests/model/test_stage_load_cpu.cpp covers the ownership.
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
  std::shared_ptr<void> keepalive; // owns the DeviceBuffer `data` points into (the lease's copies share it)
};

// "" when a Container on `device` may borrow `lease` for a text.embed_tokens of `elems` elements.
inline std::string CheckEmbedMirrorBorrow(const EmbedMirrorLease& lease, int device, size_t elems) {
  if (lease.data == nullptr || lease.elems == 0) return "the borrowed embedding mirror is empty";
  if (lease.keepalive == nullptr) return "the borrowed embedding mirror has no owner (a lease must keep its buffer alive)";
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
