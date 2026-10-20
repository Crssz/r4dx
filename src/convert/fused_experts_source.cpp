#include "fused_experts_source.hpp"

#include <algorithm>
#include <stdexcept>

namespace r4dx_convert::fused_experts {

namespace {

const std::string kGateUp = ".mlp.experts.gate_up_proj";
const std::string kDown = ".mlp.experts.down_proj";

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

[[noreturn]] void Bad(const std::string& msg) { throw std::runtime_error("fused experts source: " + msg); }

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

}  // namespace

std::shared_ptr<FusedExpertsOverlay> FusedExpertsOverlay::Create(ShardedModel& model) {
  std::shared_ptr<FusedExpertsOverlay> self(new FusedExpertsOverlay(model));
  for (const std::string& name : model.AllNames()) {
    if (!EndsWith(name, kGateUp)) continue;
    const std::string layer_prefix = name.substr(0, name.size() - kGateUp.size());  // "...layers.{i}"
    const std::string down_name = layer_prefix + kDown;
    if (!model.Has(down_name)) Bad("'" + name + "' has no sibling '" + down_name + "'");
    const TensorMeta& gm = model.RawMeta(name);
    const TensorMeta& dm = model.RawMeta(down_name);
    if (gm.dtype != "BF16" || gm.shape.size() != 3 || gm.shape[1] % 2 != 0)
      Bad("'" + name + "' is " + gm.dtype + " " + ShapeStr(gm.shape) + ", expected BF16 [E, 2 * I, H]");
    const int64_t E = gm.shape[0], I = gm.shape[1] / 2, H = gm.shape[2];
    if (dm.dtype != "BF16" || dm.shape != std::vector<int64_t>{E, H, I})
      Bad("'" + down_name + "' is " + dm.dtype + " " + ShapeStr(dm.shape) + ", expected BF16 " +
          ShapeStr({E, H, I}) + " next to " + name);
    const std::string prefix = layer_prefix + ".mlp.experts.";
    if (model.Has(prefix + "0.gate_proj.weight")) Bad("'" + prefix + "0.gate_proj.weight' exists next to '" + name + "'");
    Group g;
    g.prefix = prefix;
    g.E = E;
    g.I = I;
    g.H = H;
    const size_t gi = self->groups_.size();
    self->groups_.push_back(g);
    for (int64_t e = 0; e < E; ++e) {
      const std::string base = prefix + std::to_string(e) + ".";
      for (Kind k : {Kind::kGate, Kind::kUp, Kind::kDown}) {
        Virt v;
        v.group = gi;
        v.expert = e;
        v.kind = k;
        v.meta.dtype = "BF16";
        v.meta.shape = k == Kind::kDown ? std::vector<int64_t>{H, I} : std::vector<int64_t>{I, H};
        v.meta.begin = 0;
        v.meta.end = static_cast<uint64_t>(I) * static_cast<uint64_t>(H) * 2;
        const char* leaf = k == Kind::kGate ? "gate_proj.weight" : k == Kind::kUp ? "up_proj.weight" : "down_proj.weight";
        self->virt_.emplace(base + leaf, std::move(v));
      }
    }
  }
  return self;
}

bool FusedExpertsOverlay::Provides(const std::string& name) const { return virt_.count(name) != 0; }

const FusedExpertsOverlay::Virt& FusedExpertsOverlay::Find(const std::string& name) const {
  const auto it = virt_.find(name);
  if (it == virt_.end()) throw std::runtime_error("FusedExpertsOverlay: no virtual tensor " + name);
  return it->second;
}

const TensorMeta& FusedExpertsOverlay::Meta(const std::string& name) const { return Find(name).meta; }

const uint8_t* FusedExpertsOverlay::Base(const Virt& v) const {
  const Group& g = groups_[v.group];
  const uint64_t ih = static_cast<uint64_t>(g.I) * static_cast<uint64_t>(g.H) * 2;  // bytes of one [I, H] block
  const std::string& tensor = (v.kind == Kind::kDown ? g.prefix + "down_proj" : g.prefix + "gate_up_proj");
  const uint8_t* base = model_.RawData(tensor);
  switch (v.kind) {
    case Kind::kGate:
      return base + static_cast<uint64_t>(v.expert) * 2 * ih;
    case Kind::kUp:
      return base + static_cast<uint64_t>(v.expert) * 2 * ih + ih;
    case Kind::kDown:
      return base + static_cast<uint64_t>(v.expert) * ih;
  }
  return base;
}

const uint8_t* FusedExpertsOverlay::Data(const std::string& name) const { return Base(Find(name)); }

const uint8_t* FusedExpertsOverlay::DataRows(const std::string& name, int64_t row0, int64_t nrows,
                                             std::vector<uint8_t>& /*scratch*/) const {
  const Virt& v = Find(name);
  const int64_t N = v.meta.shape[0], K = v.meta.shape[1];
  if (row0 < 0 || nrows < 0 || row0 > N || nrows > N - row0)
    throw std::invalid_argument("fused experts DataRows: '" + name + "' rows [" + std::to_string(row0) + ", +" +
                                std::to_string(nrows) + ") outside N=" + std::to_string(N));
  return Base(v) + static_cast<uint64_t>(row0) * static_cast<uint64_t>(K) * 2;
}

void FusedExpertsOverlay::RewriteNames(std::vector<std::string>& names) const {
  names.erase(std::remove_if(names.begin(), names.end(),
                             [&](const std::string& n) {
                               for (const Group& g : groups_)
                                 if (n == g.prefix + "gate_up_proj" || n == g.prefix + "down_proj") return true;
                               return false;
                             }),
              names.end());
  names.reserve(names.size() + virt_.size());
  for (const auto& kv : virt_) names.push_back(kv.first);
}

size_t InstallFusedExpertsSource(ShardedModel& model) {
  const std::shared_ptr<FusedExpertsOverlay> overlay = FusedExpertsOverlay::Create(model);
  if (overlay->GroupCount() == 0) return 0;
  model.SetOverlay(overlay);
  return overlay->VirtualCount();
}

}  // namespace r4dx_convert::fused_experts
