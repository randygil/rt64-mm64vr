//
// RT64
//

#include "rt64_lighting_sky.h"

namespace RT64 {
    // Placeholder until the sky is implemented.
    struct LightingSky::Impl { };

    LightingSky::LightingSky(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat) {
        impl = std::make_unique<Impl>();
    }

    LightingSky::~LightingSky() { }

    bool LightingSky::enabled() const {
        return false;
    }

    void LightingSky::record(RenderWorker *worker, const LightingSkyDesc &desc) { }
};
