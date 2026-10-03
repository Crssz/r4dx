// LoadAudioEmbedder: the one audio.* tensor out of a gemma4_unified .r4dx container (header-only
// SafetensorsReader from r4dx_convert; no HIP).
#include <cstring>
#include <stdexcept>

#include "audio_embed.h"
#include "audio_frames.h"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx::audio {

bool ContainerHasAudio(const std::string& container_path) {
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(container_path));
  return reader.Has(AudioWeightName());
}

AudioEmbedder LoadAudioEmbedder(const std::string& container_path) {
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(container_path));
  const std::string name = AudioWeightName();
  if (!reader.Has(name)) {
    throw std::runtime_error(container_path + ": no '" + name +
                             "' tensor -- the container was converted without `--audio on`");
  }
  const r4dx_convert::TensorMeta& m = reader.Meta(name);
  // The converter stores a bf16 passthrough as [rows, cols, 2] raw bytes (add_bf16: shape + {2}).
  const std::vector<int64_t> want = {kEmbedHidden, kSamplesPerToken, 2};
  if (m.shape != want) {
    throw std::runtime_error(container_path + ": '" + name + "' has an unexpected shape (want [3840, 640, 2])");
  }
  std::vector<uint16_t> w(static_cast<size_t>(kEmbedHidden) * kSamplesPerToken);
  std::memcpy(w.data(), reader.Data(name), w.size() * sizeof(uint16_t));
  return AudioEmbedder(w, kSamplesPerToken, kEmbedHidden);
}

}  // namespace r4dx::audio
