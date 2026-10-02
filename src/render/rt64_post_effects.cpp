//
// RT64
//

#include "rt64_post_effects.h"

namespace RT64 {
    // Placeholder until the effects are implemented.
    struct PostEffects::Impl { };

    PostEffects::PostEffects(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat) {
        impl = std::make_unique<Impl>();
    }

    PostEffects::~PostEffects() { }

    bool PostEffects::enabled() const {
        return false;
    }

    void PostEffects::record(RenderWorker *worker, const PostEffectsSceneDesc &desc) { }
};
