# src/audio: the HOST-side half of Gemma 4 audio input (docs/gemma4-audio.md) -- WAV decode, the
# pad-to-640 / [n,640] framing, <|audio|> placeholder expansion, and the CPU RMSNorm+Linear embedder.
# CPU-only and dependency-light (no HIP, no r4dx_core, no r4dx_model), so tests/audio runs as an ordinary
# always-on unit test and the server's parse path (openai_types.cpp) can use it with no device.
add_library(r4dx_audio STATIC
  audio_wav.cpp
  audio_embed.cpp
  audio_weights.cpp
)
target_include_directories(r4dx_audio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# r4dx_convert: the header-only SafetensorsReader (LoadAudioEmbedder reads one tensor of a container).
target_link_libraries(r4dx_audio PUBLIC r4dx_convert)
