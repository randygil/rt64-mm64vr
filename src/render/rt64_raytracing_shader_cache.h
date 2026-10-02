//
// RT64
//

#pragma once

#include <unordered_map>

#include "common/rt64_plume.h"

#include "rt64_shader_common.h"
#include "rt64_shader_library.h"

namespace RT64 {
    // Only the ubershader version of the hit groups is supported for now. The map is keyed by the hash of the shader
    // description so specialized hit groups can be added later without changing the renderer.
    static const uint64_t UberShaderHash = 0;

    struct RaytracingShaderPrograms {
        RenderPipelineProgram surface;
        RenderPipelineProgram shadow;
    };

    struct RaytracingState {
        std::unique_ptr<RenderPipeline> pipeline;
        std::unordered_map<uint64_t, RaytracingShaderPrograms> shaderProgramsMap;

        // Ray generation programs in the order the renderer dispatches them:
        // 0 primary, 1 direct light, 2 indirect light, 3 reflection, 4 refraction.
        std::vector<RenderPipelineProgram> rayGenPrograms;

        // Miss programs: 0 surface, 1 shadow.
        std::vector<RenderPipelineProgram> missPrograms;
    };

    struct RaytracingShaderCache {
        static const uint32_t StateCount = 1;

        RenderDevice *device = nullptr;
        RenderShaderFormat shaderFormat = RenderShaderFormat::UNKNOWN;
        const ShaderLibrary *shaderLibrary = nullptr;
        std::unique_ptr<RenderPipelineLayout> pipelineLayout;
        std::unique_ptr<RenderShader> libraryShader;
        std::vector<uint32_t> libraryWords;
        RaytracingState states[StateCount];
        int32_t activeState = 0;
        bool setupDone = false;

        RaytracingShaderCache(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary);
        ~RaytracingShaderCache();
        bool isSetup() const;
        void setup();
        void submit(const ShaderDescription &desc);
        void setNextState();
        int32_t getActiveState() const;
    };
};
