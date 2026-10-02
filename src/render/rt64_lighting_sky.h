//
// RT64
//
// Procedural sky for the enhanced raster lighting: replaces the pixels of the background that show the sky (the game's
// 2D sky picture drawn behind the 3D scene) with a physically based atmosphere lit by the sun and clouds. It runs after
// the opaque surfaces are lit and before the translucent ones are drawn, only on the pixels nothing was drawn on.
//

#pragma once

#include <memory>

#include "common/rt64_common.h"
#include "common/rt64_plume.h"
#include "shared/rt64_lighting_params.h"

#include "rt64_render_target.h"

namespace RT64 {
    struct ShaderLibrary;

    struct LightingSkyDesc {
        // The color target is in the color write layout and must be left in it. The depth target is readable (depth
        // read layout).
        RenderTarget *colorTarget = nullptr;
        RenderTarget *depthTarget = nullptr;

        // Region of the targets covered by the scene.
        RenderRect rect;

        // Matrices, sun, camera, world basis and fog of the scene. The sky is only drawn when there's a sun.
        const interop::LightingParams *lighting = nullptr;

        // Copy of the color target before the sky is replaced (resolved if multisampled), in the shader read layout. Used
        // to tell which background pixels are sky and which are part of the game's 2D background (mountains, buildings).
        const RenderTexture *sceneColor = nullptr;

        // Seconds since the start, for the clouds.
        float time = 0.0f;
    };

    struct LightingSky {
        struct Impl;
        std::unique_ptr<Impl> impl;

        LightingSky(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat);
        ~LightingSky();

        // Whether the procedural sky is enabled, so the caller can skip copying the scene.
        bool enabled() const;

        // Replaces the sky of the region. Can change the bound framebuffer, pipeline and descriptor sets.
        void record(RenderWorker *worker, const LightingSkyDesc &desc);
    };
};
