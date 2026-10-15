// LoadLfm2TextModel: the LFM2 branch of LoadTextModel (docs/lfm2-plan.md 3.1). L1-1 placeholder: the dispatch
// reaches here for an lfm2_moe container and fails loudly; L1-8 (Lfm2LocalTextModel) and L1-10 (Lfm2TpModel)
// replace the body.
#include <memory>
#include <stdexcept>

#include "text_model.h"

namespace r4dx::model {

std::unique_ptr<TextModel> LoadLfm2TextModel(const ModelOptions& /*opts*/, const TpOptions& /*tp*/) {
  throw std::runtime_error("r4dx::model::LoadLfm2TextModel: the lfm2_moe model is not implemented yet "
                           "(docs/lfm2-plan.md L1-8 / L1-10); the container was recognised but cannot be run");
}

}  // namespace r4dx::model
