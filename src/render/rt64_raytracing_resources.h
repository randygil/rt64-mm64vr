//
// RT64
//

#pragma once

#include <memory>
#include <vector>

#include "common/rt64_common.h"
#include "common/rt64_user_configuration.h"
#include "preset/rt64_preset_scene.h"
#include "shared/rt64_interleaved_raster.h"
#include "shared/rt64_point_light.h"
#include "shared/rt64_raytracing_params.h"

#include "rt64_buffer_uploader.h"
#include "rt64_descriptor_sets.h"
#include "rt64_framebuffer_renderer_call.h"
#include "rt64_raytracing_shader_cache.h"
#include "rt64_render_target.h"
#include "rt64_render_target_manager.h"
#include "rt64_render_worker.h"
#include "rt64_shader_library.h"
#include "rt64_upscaler.h"

namespace RT64 {
    struct RaytracingConfiguration {
        int diSamples = 1;
        int giSamples = 1;
        int maxLights = 2;
        int maxReflections = 2;
        float motionBlurStrength = 0.0f;
        int motionBlurSamples = 8;
        interop::VisualizationMode visualizationMode = interop::VisualizationMode::Final;
        UpscaleMode upscalerMode = UpscaleMode::Bilinear;
        Upscaler::QualityMode upscalerQualityMode = Upscaler::QualityMode::Auto;
        float upscalerSharpness = 0.0f;
        bool upscalerResolutionOverride = false;
        float resolutionScale = 1.0f;
        bool upscalerReactiveMask = true;
        bool upscalerLockMask = false;
        bool denoiserEnabled = true;
        bool antialiasingEnabled = true;
    };

    struct RaytracingScene {
        std::vector<uint32_t> instanceIndices;
        std::vector<interop::InterleavedRaster> interleavedRasters;
        interop::float4x4 curViewMatrix;
        interop::float4x4 curProjMatrix;
        interop::float4x4 prevViewMatrix;
        interop::float4x4 prevProjMatrix;
        RenderViewport viewport;
        RenderRect scissor;
        PresetScene presetScene;
        uint32_t screenWidth = 0;
        uint32_t screenHeight = 0;
        const interop::PointLight *pointLights = nullptr;
        uint32_t lightCount = 0;
        float deltaTime = 1.0f / 30.0f;
        hlslpp::float3 worldUp = { 0.0f, 1.0f, 0.0f };
        hlslpp::float3 worldRight = { 1.0f, 0.0f, 0.0f };
        hlslpp::float3 worldForward = { 0.0f, 0.0f, 1.0f };
        hlslpp::float4 worldOrigin = { 0.0f, 0.0f, 0.0f, 0.0f };
    };

    struct RaytracingResources {
        // Maximum amount of surfaces that can be stored per pixel by the any hit shader.
        static const uint32_t MaxHitQueries = 24;

        RenderWorker *worker = nullptr;
        UserConfiguration::GraphicsAPI graphicsAPI;

        // Configuration.
        bool denoiserEnabled = true;
        int maxReflections = 2;
        float resolutionScale = 1.0f;
        UpscaleMode upscalerMode = UpscaleMode::Bilinear;
        Upscaler::QualityMode upscalerQualityMode = Upscaler::QualityMode::Auto;
        float upscalerSharpness = 0.0f;
        bool upscalerResolutionOverride = false;
        bool upscalerReactiveMask = true;
        bool upscalerLockMask = false;
        bool upscaleActive = false;
        bool antialiasingEnabled = true;

        // State.
        interop::RaytracingParams rtParams;
        BufferPair rtParamsBuffer;
        uint32_t screenWidth = 0;
        uint32_t screenHeight = 0;
        uint32_t textureWidth = 0;
        uint32_t textureHeight = 0;
        bool swapBuffers = false;
        bool skipReprojection = true;
        bool transitionOutputBuffers = true;
        bool updateOutputBuffers = true;

        // Output buffers.
        std::unique_ptr<RenderTexture> viewDirectionTexture;
        std::unique_ptr<RenderTexture> shadingPositionTexture;
        std::unique_ptr<RenderTexture> shadingNormalTexture;
        std::unique_ptr<RenderTexture> shadingSpecularTexture;
        std::unique_ptr<RenderTexture> diffuseTexture;
        std::unique_ptr<RenderTexture> instanceIdTexture;
        std::unique_ptr<RenderTexture> directLightTexture[2];
        std::unique_ptr<RenderTexture> indirectLightTexture[2];
        std::unique_ptr<RenderTexture> filteredDirectLightTexture[2];
        std::unique_ptr<RenderTexture> filteredIndirectLightTexture[2];
        std::unique_ptr<RenderTexture> reflectionTexture;
        std::unique_ptr<RenderTexture> refractionTexture;
        std::unique_ptr<RenderTexture> transparentTexture;
        std::unique_ptr<RenderTexture> flowTexture;
        std::unique_ptr<RenderTexture> reactiveMaskTexture;
        std::unique_ptr<RenderTexture> lockMaskTexture;
        std::unique_ptr<RenderTexture> normalRoughnessTexture[2];
        std::unique_ptr<RenderTexture> depthTexture[2];
        std::unique_ptr<RenderTexture> outputTexture[2];
        std::unique_ptr<RenderFramebuffer> outputFramebuffer[2];
        std::unique_ptr<RenderTexture> upscaledOutputTexture;
        std::unique_ptr<RenderTexture> antialiasedTexture[2];
        std::unique_ptr<RenderTexture> downscaledOutputTexture;
        std::unique_ptr<RenderTexture> bloomTexture[2];
        uint32_t bloomWidth = 0;
        uint32_t bloomHeight = 0;
        std::unique_ptr<RenderTexture> lumaAverageTexture;
        std::unique_ptr<RenderBuffer> lumaHistogramBuffer;
        std::unique_ptr<RenderBuffer> hitVelocityDistanceBuffer;
        std::unique_ptr<RenderBuffer> hitColorBuffer;
        std::unique_ptr<RenderBuffer> hitNormalFogBuffer;
        std::unique_ptr<RenderBuffer> hitInstanceIdBuffer;
        std::unique_ptr<RenderBufferFormattedView> hitVelocityDistanceBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitColorBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitNormalFogBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitInstanceIdBufferView;

        // Descriptor sets for the post processing passes.
        std::unique_ptr<GaussianFilterDescriptorSet> indirectFilterSets[2];
        std::unique_ptr<RaytracingComposeDescriptorSet> composeSet;
        std::unique_ptr<BicubicScalingDescriptorSet> downscaleSet;
        std::unique_ptr<LuminanceHistogramDescriptorSet> lumaSet;
        std::unique_ptr<HistogramAverageDescriptorSet> lumaAvgSet;
        std::unique_ptr<HistogramClearDescriptorSet> lumaClearSet;
        std::unique_ptr<HistogramSetDescriptorSet> lumaSetSet;
        std::unique_ptr<PostProcessDescriptorSet> postProcessSet;
        std::unique_ptr<TemporalAADescriptorSet> antialiasingSets[2];

        // Prefilter (output to bloom 0), horizontal blur (bloom 0 to 1) and vertical blur (bloom 1 to 0).
        std::unique_ptr<BloomDescriptorSet> bloomSets[3];
        bool shaderSetsDirty = true;

        // Acceleration structures.
        struct BottomLevelAS {
            RenderBottomLevelASMesh mesh;
            RenderBottomLevelASBuildInfo buildInfo;
            uint64_t bufferOffset = 0;
            uint64_t scratchOffset = 0;
            std::unique_ptr<RenderAccelerationStructure> accelerationStructure;
        };

        std::vector<BottomLevelAS> bottomLevelASVector;
        uint32_t bottomLevelASCount = 0;
        std::unique_ptr<RenderBuffer> bottomLevelASBuffer;
        uint64_t bottomLevelASBufferSize = 0;
        std::unique_ptr<RenderBuffer> scratchBuffer;
        uint64_t scratchBufferSize = 0;
        std::unique_ptr<RenderBuffer> topLevelASBuffer;
        uint64_t topLevelASBufferSize = 0;
        std::unique_ptr<RenderAccelerationStructure> topLevelAS;
        std::unique_ptr<RenderBuffer> topLevelASInstancesBuffer;
        uint64_t topLevelASInstancesBufferSize = 0;
        RenderTopLevelASBuildInfo topLevelASBuildInfo;
        std::vector<RenderTopLevelASInstance> topLevelASInstances;
        bool topLevelASPending = false;

        // Shader binding table.
        RenderShaderBindingTableInfo shaderBindingTableInfo;
        std::unique_ptr<RenderBuffer> shaderBindingTableBuffer;
        uint64_t shaderBindingTableBufferSize = 0;

        // Lights.
        std::unique_ptr<RenderBuffer> lightsBuffer;
        uint32_t lightsBufferCapacity = 0;

        // Interleaved raster targets.
        std::vector<std::unique_ptr<RenderTarget>> interleavedColorTargetVector;
        std::vector<std::unique_ptr<RenderTarget>> interleavedDepthTargetVector;
        std::vector<std::unique_ptr<RenderFramebufferStorage>> interleavedFramebufferStorageVector;
        uint32_t interleavedWidth = 0;
        uint32_t interleavedHeight = 0;
        RenderMultisampling interleavedMultisampling;
        bool interleavedUsesHDR = false;

        RaytracingResources(RenderWorker *worker, UserConfiguration::GraphicsAPI graphicsAPI);
        ~RaytracingResources();
        void setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged);
        void createOutputBuffers(RenderWorker *worker, uint32_t screenWidth, uint32_t screenHeight);
        void releaseOutputBuffers();
        void updateShaderSets(RenderWorker *worker, const ShaderLibrary *shaderLibrary);
        void updateMultisampling();
        void updateInterleavedRenderTargets(RenderWorker *worker, uint32_t width, uint32_t height, uint32_t count, const RenderMultisampling &multisampling, bool usesHDR);
        void resetBottomLevelAS();
        void addBottomLevelASMesh(const RenderBottomLevelASMesh &mesh);
        void updateBottomLevelASResources(RenderWorker *worker);
        void submitBottomLevelASCreation(RenderWorker *worker);
        void updateTopLevelASResources(RenderWorker *worker, const std::vector<InstanceDrawCall> &instanceDrawCalls, const std::vector<uint32_t> &instanceIndices);
        void submitTopLevelASCreation(RenderWorker *worker);
        void createShaderBindingTable(RenderWorker *worker, const RaytracingState *rtState, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount, const std::vector<RenderPipelineProgram> &hitGroups);
        void updateLightsBuffer(RenderWorker *worker, const RaytracingScene &rtScene);
        Upscaler *getUpscaler(UpscaleMode mode) const;
    };
};
