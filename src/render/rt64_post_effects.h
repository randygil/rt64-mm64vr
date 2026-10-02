//
// RT64
//
// Post effects applied to the 3D scene of a lit projection after its translucent surfaces are drawn and before the 2D
// elements on top of it (like the HUD): bloom, light shafts from the sun, color grading and sharpening. They only use
// the color and depth of the scene and the generic parameters of the lighting, so they work the same for any game.
//

#pragma once

#include <memory>

#include "common/rt64_common.h"
#include "common/rt64_plume.h"
#include "shared/rt64_lighting_params.h"

#include "rt64_render_target.h"

namespace RT64 {
    struct ShaderLibrary;
    struct LightingRenderer;

    struct PostEffectsSceneDesc {
        // The color target is in the color write layout and must be left in it. The depth target is readable (depth
        // read layout).
        RenderTarget *colorTarget = nullptr;
        RenderTarget *depthTarget = nullptr;

        // Region of the targets covered by the scene.
        RenderRect rect;

        // Matrices, sun, camera and fog of the scene. The effects keep their textures per scene and tell the scenes apart
        // by this address, so it must be unique to the scene within the frame and stay the same on the next frames (like
        // the parameters stored by LightingRenderer).
        const interop::LightingParams *lighting = nullptr;

        // Copy of the color target with the scene as it is when the effects start (resolved if multisampled), in the
        // shader read layout. Produced by LightingRenderer::copyColor.
        const RenderTexture *sceneColor = nullptr;

        // Seconds since the start, for animated effects.
        float time = 0.0f;
    };

    struct PostEffects {
        struct Impl;
        std::unique_ptr<Impl> impl;

        PostEffects(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat);
        ~PostEffects();

        // Whether any effect is enabled, so the caller can skip copying the scene.
        bool enabled() const;

        // Applies the effects to the region of the scene. Must be called at most once per scene and frame, and the frames
        // must be recorded one after the other (each one finishes on the GPU before the next is recorded), as a scene's
        // resources are resized and updated when it's recorded again. Can change the bound framebuffer, pipeline and
        // descriptor sets.
        void record(RenderWorker *worker, const PostEffectsSceneDesc &desc);
    };
};
