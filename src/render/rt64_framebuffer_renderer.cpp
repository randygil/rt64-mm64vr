//
// RT64
//

#include "rt64_framebuffer_renderer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

#include "../include/rt64_extended_gbi.h"

#include "shared/rt64_blender.h"

#include "common/rt64_elapsed_timer.h"
#include "common/rt64_math.h"
#include "hle/rt64_color_converter.h"
#include "gbi/rt64_f3d.h"
#include "shared/rt64_framebuffer_params.h"
#include "shared/rt64_raster_params.h"

#include "rt64_descriptor_sets.h"
#include "rt64_render_worker.h"

// TODO: Move to shared.

namespace interop {
    struct BicubicCB {
        uint2 InputResolution;
        uint2 OutputResolution;
    };

    struct HistogramAverageCB {
        uint pixelCount;
        float minLuminance;
        float luminanceRange;
        float timeDelta;
        float tau;
    };

    struct HistogramSetCB {
        float luminanceValue;
    };

    struct LuminanceHistogramCB {
        uint inputWidth;
        uint inputHeight;
        float minLuminance;
        float oneOverLuminanceRange;
    };

    struct TextureCB {
        uint2 TextureSize;
        float2 TexelSize;
    };

    struct TemporalAACB {
        uint2 TextureSize;
        float2 TexelSize;
        float BlendFactor;
        uint Reset;
    };

    struct BloomCB {
        uint2 OutputSize;
        float2 InputTexelSize;
        uint Mode;
        float Threshold;
    };
};

namespace RT64 {
    // Helper functions.

    static std::atomic<float> enhancementIntensity = 1.0f;

    void setEnhancementIntensity(float intensity) {
        enhancementIntensity = std::clamp(intensity, 0.0f, 1.0f);
    }

    float getEnhancementIntensity() {
        return enhancementIntensity;
    }

    static std::atomic<bool> proceduralSkyEnabled = true;

#if RT_ENABLED
    // Same single scattering model the sky shader uses, so the bounces can use a few precomputed colors of the sky
    // instead of evaluating it for every ray.
    static hlslpp::float3 atmosphereScattering(hlslpp::float3 direction, hlslpp::float3 sunDirection) {
        const float PlanetRadius = 6371e3f;
        const float AtmosphereRadius = 6471e3f;
        const hlslpp::float3 RayleighCoefficient = { 5.5e-6f, 13.0e-6f, 22.4e-6f };
        const float MieCoefficient = 21e-6f;
        const float RayleighHeight = 8e3f;
        const float MieHeight = 1.2e3f;
        const float MieAnisotropy = 0.758f;
        const int PrimarySteps = 16;
        const int LightSteps = 8;
        auto sphereExit = [](hlslpp::float3 origin, hlslpp::float3 dir, float radius) {
            const float b = float(hlslpp::dot(dir, origin));
            const float c = float(hlslpp::dot(origin, origin)) - radius * radius;
            const float d = b * b - c;
            return (d < 0.0f) ? 0.0f : (-b + sqrtf(d));
        };

        const hlslpp::float3 origin = { 0.0f, PlanetRadius + 200.0f, 0.0f };
        const float stepSize = sphereExit(origin, direction, AtmosphereRadius) / float(PrimarySteps);
        hlslpp::float3 totalRayleigh = { 0.0f, 0.0f, 0.0f };
        hlslpp::float3 totalMie = { 0.0f, 0.0f, 0.0f };
        float depthRayleigh = 0.0f;
        float depthMie = 0.0f;
        for (int i = 0; i < PrimarySteps; i++) {
            const hlslpp::float3 position = origin + direction * (stepSize * (float(i) + 0.5f));
            const float height = float(hlslpp::length(position)) - PlanetRadius;
            const float stepRayleigh = expf(-height / RayleighHeight) * stepSize;
            const float stepMie = expf(-height / MieHeight) * stepSize;
            depthRayleigh += stepRayleigh;
            depthMie += stepMie;

            const float lightStepSize = sphereExit(position, sunDirection, AtmosphereRadius) / float(LightSteps);
            float lightRayleigh = 0.0f;
            float lightMie = 0.0f;
            for (int j = 0; j < LightSteps; j++) {
                const hlslpp::float3 lightPosition = position + sunDirection * (lightStepSize * (float(j) + 0.5f));
                const float lightHeight = std::max(float(hlslpp::length(lightPosition)) - PlanetRadius, 0.0f);
                lightRayleigh += expf(-lightHeight / RayleighHeight) * lightStepSize;
                lightMie += expf(-lightHeight / MieHeight) * lightStepSize;
            }

            const hlslpp::float3 opticalDepth = RayleighCoefficient * (depthRayleigh + lightRayleigh) + MieCoefficient * (depthMie + lightMie);
            const hlslpp::float3 attenuation = { expf(-float(opticalDepth.x)), expf(-float(opticalDepth.y)), expf(-float(opticalDepth.z)) };
            totalRayleigh += attenuation * stepRayleigh;
            totalMie += attenuation * stepMie;
        }

        const float Pi = 3.14159265f;
        const float mu = float(hlslpp::dot(direction, sunDirection));
        const float g2 = MieAnisotropy * MieAnisotropy;
        const float phaseRayleigh = 3.0f / (16.0f * Pi) * (1.0f + mu * mu);
        const float phaseMie = 3.0f / (8.0f * Pi) * ((1.0f - g2) * (mu * mu + 1.0f)) / (powf(std::max(1.0f + g2 - 2.0f * mu * MieAnisotropy, 1e-6f), 1.5f) * (2.0f + g2));
        return (RayleighCoefficient * totalRayleigh * phaseRayleigh + totalMie * (MieCoefficient * phaseMie)) * 22.0f;
    }
#endif

    void setProceduralSkyEnabled(bool enabled) {
        proceduralSkyEnabled = enabled;
    }

    bool isProceduralSkyEnabled() {
        return proceduralSkyEnabled;
    }

    // Values of the visual enhancements. They can be overridden with environment variables or with the file pointed
    // by RT64_RT_TUNING_FILE, which uses one "NAME value" pair per line and is reloaded whenever it changes.
    float enhancementValue(const char *name, float defaultValue) {
        static std::mutex tuningMutex;
        std::scoped_lock lock(tuningMutex);
        static std::unordered_map<std::string, float> tuningValues;
        static std::filesystem::file_time_type tuningTime;
        static uint32_t tuningCheckCounter = 0;
        static const char *tuningPath = getenv("RT64_RT_TUNING_FILE");
        if ((tuningPath != nullptr) && ((tuningCheckCounter++ % 256) == 0)) {
            std::error_code ec;
            const auto writeTime = std::filesystem::last_write_time(tuningPath, ec);
            if (!ec && (writeTime != tuningTime)) {
                tuningTime = writeTime;
                tuningValues.clear();
                std::ifstream tuningStream(tuningPath);
                std::string key;
                float value;
                while (tuningStream >> key >> value) {
                    tuningValues[key] = value;
                }
            }
        }

        auto it = tuningValues.find(name);
        if (it != tuningValues.end()) {
            return it->second;
        }

        const char *value = getenv(name);
        return (value != nullptr) ? float(atof(value)) : defaultValue;
    }

    struct GPUMarkerState {
        static const uint32_t MaxMarkers = 128;
        RenderQueryPool *queryPool = nullptr;
        RenderQueryPool *pendingQueryPool = nullptr;
        uint32_t count = 0;
        uint32_t pendingCount = 0;
        const char *names[MaxMarkers] = {};
        std::vector<std::pair<std::string, double>> totals;
        uint32_t frames = 0;
    };

    static GPUMarkerState gpuMarkerState;

    void gpuMarker(RenderCommandList *commandList, const char *name) {
        GPUMarkerState &s = gpuMarkerState;
        if ((s.queryPool == nullptr) || (s.count >= GPUMarkerState::MaxMarkers)) {
            return;
        }

        commandList->writeTimestamp(s.queryPool, s.count);
        s.names[s.count++] = name;
    }

    void beginGPUMarkers(RenderCommandList *commandList, RenderQueryPool *queryPool) {
        GPUMarkerState &s = gpuMarkerState;
        s.queryPool = queryPool;
        s.count = 0;
        commandList->resetQueryPool(queryPool, 0, GPUMarkerState::MaxMarkers);
        gpuMarker(commandList, "start");
    }

    void endGPUMarkers(RenderCommandList *commandList) {
        GPUMarkerState &s = gpuMarkerState;
        if (s.queryPool == nullptr) {
            return;
        }

        gpuMarker(commandList, "end of frame");

        // Vulkan only returns the results when all the queries of the pool were written.
        for (uint32_t i = s.count; i < GPUMarkerState::MaxMarkers; i++) {
            commandList->writeTimestamp(s.queryPool, i);
        }

        s.pendingQueryPool = s.queryPool;
        s.pendingCount = s.count;
        s.queryPool = nullptr;
    }

    void readGPUMarkers() {
        GPUMarkerState &s = gpuMarkerState;
        if (s.pendingQueryPool == nullptr) {
            return;
        }

        s.pendingQueryPool->queryResults();
        const uint64_t *results = s.pendingQueryPool->getResults();
        for (uint32_t i = 1; i < s.pendingCount; i++) {
            const double ms = double(results[i] - results[i - 1]) / 1000000.0;
            auto it = std::find_if(s.totals.begin(), s.totals.end(), [&](const auto &total) { return total.first == s.names[i]; });
            if (it != s.totals.end()) {
                it->second += ms;
            }
            else {
                s.totals.emplace_back(s.names[i], ms);
            }
        }

        s.frames++;
        s.pendingQueryPool = nullptr;
    }

    void printGPUMarkers() {
        GPUMarkerState &s = gpuMarkerState;
        if (s.frames == 0) {
            return;
        }

        std::string line = "GPU passes (ms):";
        char buffer[128];
        for (const auto &total : s.totals) {
            snprintf(buffer, sizeof(buffer), " %s %.3f,", total.first.c_str(), total.second / s.frames);
            line += buffer;
        }

        fprintf(stderr, "%s\n", line.c_str());
        s.totals.clear();
        s.frames = 0;
    }
    
    RenderRect convertFixedRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, int32_t horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        if (!rect.isNull()) {
            auto computeOrigin = [=](uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return std::lround(((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage));
                }
                else {
                    return fbWidth / 2L;
                }
            };

            auto correctMisalignment = [=](int32_t coord, uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return int32_t(coord - (coord % std::lround(resScale[1]))) - horizontalMisalignment;
                }
                else {
                    return coord;
                }
            };

            int32_t left = static_cast<int32_t>(std::floor((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x));
            int32_t right = static_cast<int32_t>(std::ceil((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x));
            int32_t top = lround(rect.top(true) * resScale.y);
            int32_t bottom = lround(rect.bottom(true) * resScale.y);
            left = correctMisalignment(left, leftOrigin);
            right = correctMisalignment(right, rightOrigin);
            return RenderRect(left, top, right, bottom);
        }
        else {
            return RenderRect(0, 0, 0, 0);
        }
    }
    
    RenderViewport convertViewportRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, float horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        auto computeOrigin = [=](uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return ((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage);
            }
            else {
                return float(fbWidth) / 2;
            }
        };
        
        auto correctMisalignment = [=](float coord, uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return (coord - std::fmod(coord, resScale[1])) - horizontalMisalignment;
            }
            else {
                return coord;
            }
        };
        
        float left = std::round((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x);
        float right = std::round((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x);
        float top = std::round(rect.top(true) * resScale.y);
        float bottom = std::round(rect.bottom(true) * resScale.y);
        left = correctMisalignment(left, leftOrigin);
        right = correctMisalignment(right, rightOrigin);
        return RenderViewport(left, top, right - left, bottom - top);
    }

    hlslpp::float3 viewPositionFrom(hlslpp::float4x4 viewI) {
        return viewI[3].xyz;
    }

    hlslpp::float3 viewDirectionFrom(hlslpp::float4x4 viewI) {
        return hlslpp::normalize(viewI[2].xyz);
    }

    RenderColor toRenderColor(hlslpp::float4 v) {
        return { v.x, v.y, v.z, v.w };
    }

    static RenderRect viewportScissorIntersection(const RenderViewport &viewport, const RenderRect &scissor) {
        return RenderRect{
            std::max(static_cast<int32_t>(std::floor(viewport.x)), scissor.left),
            std::max(static_cast<int32_t>(std::floor(viewport.y)), scissor.top),
            std::min(static_cast<int32_t>(std::ceil(viewport.x + viewport.width)), scissor.right),
            std::min(static_cast<int32_t>(std::ceil(viewport.y + viewport.height)), scissor.bottom)
        };
    }

    // RasterScene

    RasterScene::RasterScene() { }

    // FramebufferRenderer
    
    FramebufferRenderer::FramebufferRenderer(RenderWorker *worker, bool rtSupport, UserConfiguration::GraphicsAPI graphicsAPI, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);

        this->shaderLibrary = shaderLibrary;

        frameParams.frameCount = 0;
        frameParams.viewUbershaders = false;
        frameParams.ditherNoiseStrength = 1.0f;
        frameParams.cutoutAntialiasing = 0;

        shaderUploader = std::make_unique<BufferUploader>(worker->device);
        descCommonSet = std::make_unique<FramebufferRendererDescriptorCommonSet>(shaderLibrary->samplerLibrary, worker->device->getCapabilities().raytracing, worker->device);

        // Only the main renderer (the one that supports raytracing) draws the enhanced lighting.
        if (rtSupport) {
            RenderShaderFormat shaderFormat = RenderShaderFormat::SPIRV;
            if (graphicsAPI == UserConfiguration::GraphicsAPI::D3D12) {
                shaderFormat = RenderShaderFormat::DXIL;
            }
            else if (graphicsAPI == UserConfiguration::GraphicsAPI::Metal) {
                shaderFormat = RenderShaderFormat::METAL;
            }

            lighting = std::make_unique<LightingRenderer>(worker->device, shaderLibrary, shaderFormat);
        }

#   if RT_ENABLED
        if (rtSupport) {
            this->rtSupport = rtSupport;
            rtResources = std::make_unique<RaytracingResources>(worker, graphicsAPI);
    }
#   endif
    }

    FramebufferRenderer::~FramebufferRenderer() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
    }
    
    void FramebufferRenderer::resetFramebuffers(RenderWorker *worker, bool ubershadersVisible, float ditherNoiseStrength, const RenderMultisampling &multisampling) {
        instanceDrawCallVector.clear();
        hitGroupVector.clear();
        renderIndicesVector.clear();
        rspSmoothNormalVector.clear();
        if (lighting != nullptr) {
            lighting->reset();
        }

        frameParams.viewUbershaders = ubershadersVisible;
        frameParams.ditherNoiseStrength = ditherNoiseStrength;
        frameParams.cutoutAntialiasing = ((lighting != nullptr) && isRasterLightingEnabled() && (enhancementValue("RT64_LIGHT_CUTOUT_AA", 1.0f) > 0.0f)) ? 1 : 0;
        framebufferCount = 0;

        // Create dummy color target if it hasn't been created yet.
        if (dummyColorTarget == nullptr) {
            RenderTextureDesc dummyColorDesc = RenderTextureDesc::ColorTarget(4, 4, RenderTarget::colorBufferFormat(shaderLibrary->usesHDR), multisampling);
            dummyColorTarget = worker->device->createTexture(dummyColorDesc);
            dummyColorTarget->setName("Framebuffer Renderer Color Dummy");
            dummyColorTargetView = dummyColorTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyColorDesc.format));
            dummyColorTargetTransitioned = false;
        }

        // Create dummy depth target if it hasn't been created yet.
        if (dummyDepthTarget == nullptr) {
            RenderTextureDesc dummyDepthDesc = RenderTextureDesc::DepthTarget(4, 4, RenderFormat::D32_FLOAT, multisampling);
            dummyDepthTarget = worker->device->createTexture(dummyDepthDesc);
            dummyDepthTarget->setName("Framebuffer Renderer Depth Dummy");
            dummyDepthTargetView = dummyDepthTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyDepthDesc.format));
            dummyDepthTargetTransitioned = false;
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::resetRaytracing(RaytracingShaderCache *rtShaderCache, const RenderTexture *blueNoiseTexture) {
        assert(rtResources != nullptr);
        assert(rtShaderCache != nullptr);
        assert(blueNoiseTexture != nullptr);

        const int32_t stateIndex = rtShaderCache->getActiveState();
        rtState = &rtShaderCache->states[stateIndex];
        rtPipelineLayout = rtShaderCache->pipelineLayout.get();
        rtResources->resetBottomLevelAS();

        this->blueNoiseTexture = blueNoiseTexture;
    }
#endif

    void FramebufferRenderer::updateTextureCache(TextureCache *textureCache) {
        const std::unique_lock<std::mutex> textureMapLock(textureCache->textureMapMutex);
        textureCacheVersions = textureCache->textureMap.versions;
        textureCacheTextures = textureCache->textureMap.textures;
        textureCacheTextureReplacements = textureCache->textureMap.textureReplacements;
        textureCacheFreeSpaces = textureCache->textureMap.freeSpaces;
        textureCacheSize = static_cast<uint32_t>(textureCacheTextures.size());
        textureCacheGlobalVersion = textureCache->textureMap.globalVersion;
        textureCacheReplacementMapEnabled = textureCache->textureMap.replacementMapEnabled;
        dynamicTextureViewVector.clear();
        dynamicTextureBarrierVector.clear();
    }

    void FramebufferRenderer::createGPUTiles(const DrawCallTile *callTiles, uint32_t callTileCount, interop::GPUTile *dstGPUTiles, const FramebufferManager *fbManager, 
        TextureCache *textureCache, uint64_t submissionFrame)
    {
        for (uint32_t i = 0; i < callTileCount; i++) {
            const DrawCallTile &callTile = callTiles[i];
            if (!callTile.valid) {
                continue;
            }
            
            interop::GPUTile &gpuTile = dstGPUTiles[i];
            if (callTile.tileCopyUsed) {
                const auto &it = fbManager->tileCopies.find(callTile.tmemHashOrID);
                if (it != fbManager->tileCopies.end()) {
                    const FramebufferManager::TileCopy &tileCopy = it->second;
                    gpuTile.tcScale.x = static_cast<float>(tileCopy.usedWidth) / static_cast<float>(callTile.tileCopyWidth);
                    gpuTile.tcScale.y = static_cast<float>(tileCopy.usedHeight) / static_cast<float>(callTile.tileCopyHeight);
                    gpuTile.ulScale.x = tileCopy.ulScaleS ? gpuTile.tcScale.x : 1.0f;
                    gpuTile.ulScale.y = tileCopy.ulScaleT ? gpuTile.tcScale.y : 1.0f;
                    gpuTile.texelShift = tileCopy.texelShift;
                    gpuTile.texelMask = tileCopy.texelMask;
                    gpuTile.textureIndex = getTextureIndex(tileCopy);
                    gpuTile.textureDimensions = interop::float3(float(tileCopy.textureWidth), float(tileCopy.textureHeight), 1.0f);
                    gpuTile.flags.alphaIsCvg = !callTile.reinterpretTile;
                    gpuTile.flags.highRes = true;
                    gpuTile.flags.fromCopy = true;
                    gpuTile.flags.rawTMEM = false;
                    gpuTile.flags.hasMipmaps = false;
                }
            }
            else {
                // Retrieve the texture from the cache or use a blank texture if not found.
                uint32_t textureIndex = 0;
                bool textureReplaced = false;
                bool hasMipmaps = false;
                bool shiftedByHalf = false;
                textureCache->useTexture(callTile.tmemHashOrID, submissionFrame, textureIndex, gpuTile.tcScale, gpuTile.textureDimensions, textureReplaced, hasMipmaps, shiftedByHalf);

                // Describe the GPU tile for a regular texture.
                gpuTile.ulScale.x = gpuTile.tcScale.x;
                gpuTile.ulScale.y = gpuTile.tcScale.y;
                gpuTile.texelShift = { 0, 0 };
                gpuTile.texelMask = { UINT_MAX, UINT_MAX };
                gpuTile.textureIndex = textureIndex;
                gpuTile.flags.alphaIsCvg = false;
                gpuTile.flags.highRes = textureReplaced;
                gpuTile.flags.fromCopy = false;
                gpuTile.flags.rawTMEM = !textureReplaced && callTile.rawTMEM;
                gpuTile.flags.hasMipmaps = hasMipmaps;
                gpuTile.flags.shiftedByHalf = shiftedByHalf;
            }
        }
    }

    uint32_t FramebufferRenderer::getDestinationIndex() {
        uint32_t dstIndex;
        if (textureCacheFreeSpaces.empty()) {
            dstIndex = textureCacheSize++;
        }
        else {
            dstIndex = textureCacheFreeSpaces.back();
            textureCacheFreeSpaces.pop_back();
        }

        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(RenderTarget *renderTarget) {
        assert(renderTarget != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ renderTarget->getResolvedTexture(), dstIndex, renderTarget->getResolvedTextureView()});
        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(const FramebufferManager::TileCopy &tileCopy) {
        assert(tileCopy.texture != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ tileCopy.texture.get(), dstIndex, nullptr });
        dynamicTextureBarrierVector.emplace_back(RenderTextureBarrier(tileCopy.texture.get(), RenderTextureLayout::SHADER_READ));
        return dstIndex;
    }
    
    void FramebufferRenderer::updateShaderDescriptorSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        assert(worker != nullptr);
        assert(drawBuffers != nullptr);
        
        const bool createSet = (descTextureSet == nullptr) || (descTextureSet->textureCacheSize < (textureCacheSize + 1));
        if (createSet) {
            descTextureSet = std::make_unique<FramebufferRendererDescriptorTextureSet>(worker->device, ((textureCacheSize + 1) * 3) / 2);
        }

        if (createSet || (descriptorTextureReplacementMapEnabled != textureCacheReplacementMapEnabled)) {
            descriptorTextureVersions.clear();
            descriptorTextureGlobalVersion = 0;
            descriptorTextureReplacementMapEnabled = textureCacheReplacementMapEnabled;
        }

#   if RT_ENABLED
        if (raytracingEnabled) {
            // The barrier interface requires mutable buffers even though they're only read from.
            const RenderBuffer *inputBuffers[] = {
                outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldVelBuffer.buffer.get(),
                outputBuffers->genTexCoordBuffer.buffer.get(), outputBuffers->shadedColBuffer.buffer.get(), drawBuffers->fogIndicesBuffer.get(),
                drawBuffers->lightIndicesBuffer.get(), drawBuffers->lightCountsBuffer.get(), drawBuffers->faceIndicesBuffer.get()
            };

            rtInputBuffers.clear();
            for (const RenderBuffer *buffer : inputBuffers) {
                rtInputBuffers.emplace_back(const_cast<RenderBuffer *>(buffer));
            }

            descCommonSet->setBuffer(descCommonSet->posBuffer, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->normBuffer, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->velBuffer, outputBuffers->worldVelBuffer.buffer.get(), outputBuffers->worldVelBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->genTexCoordBuffer, outputBuffers->genTexCoordBuffer.buffer.get(), outputBuffers->genTexCoordBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->shadedColBuffer, outputBuffers->shadedColBuffer.buffer.get(), outputBuffers->shadedColBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcFogIndices, drawBuffers->fogIndicesBuffer.get(), drawBuffers->fogIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightIndices, drawBuffers->lightIndicesBuffer.get(), drawBuffers->lightIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightCounts, drawBuffers->lightCountsBuffer.get(), drawBuffers->lightCountsBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->indexBuffer, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->RSPFogVector, drawBuffers->rspFogBuffer.get(), drawBuffers->rspFogBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPFog)));
            descCommonSet->setBuffer(descCommonSet->RSPLightVector, drawBuffers->rspLightsBuffer.get(), drawBuffers->rspLightsBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPLight)));
            descCommonSet->setTexture(descCommonSet->gViewDirection, rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingPosition, rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingNormal, rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingSpecular, rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDiffuse, rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gInstanceId, rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReflection, rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gRefraction, rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gTransparent, rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFlow, rtResources->flowTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReactiveMask, rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gLockMask, rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDepth, rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDepth, rtResources->depthTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredDirectLight, rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredIndirectLight, rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL);

            const uint32_t hitBufferPixelCount = rtResources->textureWidth * rtResources->textureHeight * RaytracingResources::MaxHitQueries;
            descCommonSet->setBuffer(descCommonSet->gHitVelocityDistance, rtResources->hitVelocityDistanceBuffer.get(), hitBufferPixelCount * 16, rtResources->hitVelocityDistanceBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitColor, rtResources->hitColorBuffer.get(), hitBufferPixelCount * 4, rtResources->hitColorBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitNormalFog, rtResources->hitNormalFogBuffer.get(), hitBufferPixelCount * 8, rtResources->hitNormalFogBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitInstanceId, rtResources->hitInstanceIdBuffer.get(), hitBufferPixelCount * 2, rtResources->hitInstanceIdBufferView.get());
            descCommonSet->setAccelerationStructure(descCommonSet->SceneBVH, rtResources->topLevelAS.get());
            descCommonSet->setBuffer(descCommonSet->SceneLights, rtResources->lightsBuffer.get(), sizeof(interop::PointLight) * std::max(rtResources->rtParams.lightsCount, 1U), RenderBufferStructuredView(sizeof(interop::PointLight)));
            descCommonSet->setBuffer(descCommonSet->interleavedRasters, interleavedRastersBuffer.get(), sizeof(interop::InterleavedRaster) * std::max(interleavedRastersCount, 1U), RenderBufferStructuredView(sizeof(interop::InterleavedRaster)));
            descCommonSet->setTexture(descCommonSet->gBlueNoise, blueNoiseTexture, RenderTextureLayout::SHADER_READ);
            descCommonSet->setBuffer(descCommonSet->instanceExtraParams, drawBuffers->extraParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::ExtraParams)));
            descCommonSet->setBuffer(descCommonSet->RtParams, rtResources->rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        }
#   endif

        // The passes of the enhanced lighting read the world positions, normals and indices of the vertices.
        if (lightingBuffersActive && !raytracingEnabled) {
            descCommonSet->setBuffer(descCommonSet->posBuffer, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->normBuffer, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->indexBuffer, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize);
        }

        descCommonSet->setBuffer(descCommonSet->FrParams, frameParamsBuffer.get(), sizeof(interop::FrameParams));
        descCommonSet->setBuffer(descCommonSet->instanceRenderIndices, renderIndicesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderIndices)));
        descCommonSet->setBuffer(descCommonSet->instanceRDPParams, drawBuffers->rdpParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPParams)));
        descCommonSet->setBuffer(descCommonSet->RDPTiles, drawBuffers->rdpTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPTile)));
        descCommonSet->setBuffer(descCommonSet->GPUTiles, drawBuffers->gpuTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::GPUTile)));
        descCommonSet->setBuffer(descCommonSet->DynamicRenderParams, drawBuffers->renderParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderParams)));

        // Make sure the versions vector matches the texture cache size.
        descriptorTextureVersions.resize(textureCacheSize, 0);

        // Update texture vector with static textures from the cache and dynamic resource views.
        if (descriptorTextureGlobalVersion != textureCacheGlobalVersion) {
            const uint32_t textureVersionSize = static_cast<uint32_t>(textureCacheVersions.size());
            for (uint32_t i = 0; i < textureVersionSize; i++) {
                if (textureCacheVersions[i] == descriptorTextureVersions[i]) {
                    continue;
                }

                descriptorTextureVersions[i] = textureCacheVersions[i];
                if (textureCacheTextures[i] == nullptr) {
                    continue;
                }

                if (textureCacheReplacementMapEnabled && (textureCacheTextureReplacements[i] != nullptr)) {
                    descTextureSet->setTexture(i, textureCacheTextureReplacements[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else if (textureCacheTextures[i]->texture != nullptr) {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->tmem.get(), RenderTextureLayout::SHADER_READ);
                }
            }

            descriptorTextureGlobalVersion = textureCacheGlobalVersion;
        }

        for (const DynamicTextureView &dynamicView : dynamicTextureViewVector) {
            descTextureSet->setTexture(dynamicView.dstIndex, dynamicView.texture, RenderTextureLayout::SHADER_READ, dynamicView.textureView);
            descriptorTextureVersions[dynamicView.dstIndex] = 0;
        }
    }

    void FramebufferRenderer::updateRSPSmoothNormalSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (smoothDescSet == nullptr) {
            smoothDescSet = std::make_unique<RSPSmoothNormalDescriptorSet>(worker->device);
        }

        smoothDescSet->setBuffer(smoothDescSet->srcWorldPos, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcCol, drawBuffers->normalColorBuffer.get(), drawBuffers->normalColorBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint8_t) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        smoothDescSet->setBuffer(smoothDescSet->dstWorldNorm, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
        if (smoothNormalGroupsBuffer.get() != nullptr) {
            smoothDescSet->setBuffer(smoothDescSet->srcGroups, smoothNormalGroupsBuffer.get(), RenderBufferStructuredView(sizeof(interop::uint4)));
        }
    }

    void FramebufferRenderer::updateRSPVertexTestZSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (vertexTestZSet == nullptr) {
            vertexTestZSet = std::make_unique<RSPVertexTestZDescriptorSet>(worker->device);
        }
        
        vertexTestZSet->setBuffer(vertexTestZSet->screenPos, outputBuffers->screenPosBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 4));
        vertexTestZSet->setBuffer(vertexTestZSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        vertexTestZSet->setBuffer(vertexTestZSet->dstFaceIndices, outputBuffers->testZIndexBuffer.buffer.get(), outputBuffers->testZIndexBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
    }

    void FramebufferRenderer::updateShaderViews(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        updateShaderDescriptorSet(worker, drawBuffers, outputBuffers, raytracingEnabled);
        updateRSPVertexTestZSet(worker, drawBuffers, outputBuffers);

        if (lightingBuffersActive && !raytracingEnabled) {
            updateRSPSmoothNormalSet(worker, drawBuffers, outputBuffers);
        }

#   if RT_ENABLED
        if (raytracingEnabled) {
            updateRSPSmoothNormalSet(worker, drawBuffers, outputBuffers);
            rtResources->updateShaderSets(worker, shaderLibrary);
        }
#   endif
    }

    bool FramebufferRenderer::submitDepthAccess(RenderWorker *worker, RenderFramebufferStorage *fbStorage, bool readOnly, bool &depthState) {
        if (depthState == readOnly) {
            return false;
        }
        
        RenderFramebuffer *renderFramebuffer = readOnly ? fbStorage->colorWriteDepthRead.get() : fbStorage->colorDepthWrite.get();
        const RenderTextureLayout depthReadState = RenderTextureLayout::DEPTH_READ;
        const RenderTextureLayout depthWriteState = RenderTextureLayout::DEPTH_WRITE;
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(fbStorage->depthTarget->texture.get(), readOnly ? depthReadState : RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(renderFramebuffer);
        depthState = readOnly;
        return true;
    }
    
    void FramebufferRenderer::submitRasterScene(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState) {
        InstanceDrawCall::Type previousCallType = InstanceDrawCall::Type::Unknown;
        bool previousVertexTestZ = false;
        const RenderPipeline *previousPipeline = nullptr;
        RenderRect previousScissor;
        interop::RasterParams rasterParams;
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        RenderDescriptorSet *descDummyFbSet = framebuffer.descDummyFbSet->get();

        auto switchToGraphicsPipeline = [&]() {
            previousCallType = InstanceDrawCall::Type::Unknown;
            previousVertexTestZ = false;
            previousPipeline = nullptr;
            previousScissor = RenderRect();
            worker->commandList->setGraphicsPipelineLayout(rendererPipelineLayout);
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(depthState ? descRealFbSet : descDummyFbSet, 3);
            worker->commandList->setViewports(framebuffer.viewport);
        };

        auto switchToDepthRead = [&]() {
            if (submitDepthAccess(worker, fbStorage, true, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            }
        };

        auto switchToDepthWrite = [&]() {
            if (submitDepthAccess(worker, fbStorage, false, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descDummyFbSet, 3);
            }
        };

        auto drawCallTriangles = [&](const InstanceDrawCall &drawCall) {
            if (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) {
                worker->commandList->drawIndexedInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0, 0);
            }
            else {
                worker->commandList->drawInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0);
            }
        };

        if (fbStorage->colorTarget != nullptr) {
            switchToGraphicsPipeline();
        }
        
        for (uint32_t i : rasterScene.instanceIndices) {
            const InstanceDrawCall &drawCall = instanceDrawCallVector[i];
            switch (drawCall.type) {
            case InstanceDrawCall::Type::IndexedTriangles: 
            case InstanceDrawCall::Type::RawTriangles:
            case InstanceDrawCall::Type::RegularRect: {
                assert(fbStorage->colorTarget != nullptr);

                const bool typeDifferent = (drawCall.type != previousCallType);
                const bool testZDifferent = (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) && (drawCall.triangles.vertexTestZ != previousVertexTestZ);
                if (typeDifferent || testZDifferent) {
                    switch (drawCall.type) {
                    case InstanceDrawCall::Type::IndexedTriangles:
                        worker->commandList->setVertexBuffers(0, indexedVertexViews.data(), uint32_t(indexedVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(drawCall.triangles.vertexTestZ ? &testZIndexBufferView : &indexBufferView);
                        previousVertexTestZ = drawCall.triangles.vertexTestZ;
                        break;
                    case InstanceDrawCall::Type::RawTriangles:
                    case InstanceDrawCall::Type::RegularRect:
                        worker->commandList->setVertexBuffers(0, rawVertexViews.data(), uint32_t(rawVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(nullptr);
                        break;
                    default:
                        assert(false && "Unknown draw call type.");
                        break;
                    };

                    previousCallType = drawCall.type;
                }

                const auto &triangles = drawCall.triangles;
                assert(triangles.pipeline != nullptr);

                // Draw calls can sometimes end up with empty scissors and cause validation errors. We just skip them.
                if (triangles.scissor.isEmpty()) {
                    continue;
                }

                // A new pass must be started if decals are required and something wrote to the depth buffer before this call.
                const interop::OtherMode otherMode = triangles.shaderDesc.otherMode;
                bool depthDecal = (otherMode.zMode() == ZMODE_DEC);
                bool depthWrite = otherMode.zUpd();
                if (depthDecal) {
                    switchToDepthRead();
                }
                else if (!depthDecal && depthWrite) {
                    switchToDepthWrite();
                }

                if (previousScissor != triangles.scissor) {
                    worker->commandList->setScissors(triangles.scissor);
                    previousScissor = triangles.scissor;
                }

                if (previousPipeline != triangles.pipeline) {
                    worker->commandList->setPipeline(triangles.pipeline);
                    previousPipeline = triangles.pipeline;
                }
                
                rasterParams.renderIndex = i;
                rasterParams.screenScale = triangles.screenScale;
                rasterParams.screenOffset = triangles.screenOffset;
                worker->commandList->setGraphicsPushConstants(0, &rasterParams);

                drawCallTriangles(drawCall);

                // Simulate dither noise.
                if (triangles.postBlendDitherNoise) {
                    if (triangles.postBlendDitherNoiseNegative) {
                        worker->commandList->setPipeline(postBlendDitherNoiseSubNegativePipeline);
                    }
                    else {
                        worker->commandList->setPipeline(postBlendDitherNoiseAddPipeline);
                        drawCallTriangles(drawCall);

                        worker->commandList->setPipeline(postBlendDitherNoiseSubPipeline);
                    }

                    drawCallTriangles(drawCall);
                    previousPipeline = nullptr;
                }

                break;
            };
            case InstanceDrawCall::Type::FillRect: {
                const auto &clearRect = drawCall.clearRect;
                RenderTarget *chosenTarget = (fbStorage->colorTarget != nullptr) ? fbStorage->colorTarget : fbStorage->depthTarget;
                bool rectCoversWholeTarget = (clearRect.rect.left == 0) && (clearRect.rect.top == 0) && (uint32_t(clearRect.rect.right) == chosenTarget->width) && (uint32_t(clearRect.rect.bottom) == chosenTarget->height);
                const RenderRect *clearRects = rectCoversWholeTarget ? nullptr : &clearRect.rect;
                uint32_t clearRectCount = rectCoversWholeTarget ? 0 : 1;
                if (fbStorage->colorTarget != nullptr) {
                    worker->commandList->clearColor(0, clearRect.color, clearRects, clearRectCount);
                }
                else {
                    worker->commandList->clearDepth(true, clearRect.depth, clearRects, clearRectCount);
                }

                break;
            };
            case InstanceDrawCall::Type::Lighting: {
                if ((lighting != nullptr) && (fbStorage->colorTarget != nullptr)) {
                    // The lighting reads the depth buffer, so it's switched to the read only layout first.
                    gpuMarker(worker->commandList.get(), "raster");
                    submitDepthAccess(worker, fbStorage, true, depthState);
                    gpuMarker(worker->commandList.get(), "depth to read");
                    lighting->recordGBuffer(worker, drawCall.lighting.sceneIndex, descCommonSet->get(), descTextureSet->get(), descRealFbSet, indexedVertexViews.data(),
                        vertexInputSlots.data(), uint32_t(indexedVertexViews.size()), &indexBufferView, instanceDrawCallVector);
                    gpuMarker(worker->commandList.get(), "gbuffer");
                    lighting->recordAmbientOcclusion(worker, drawCall.lighting.sceneIndex);
                    gpuMarker(worker->commandList.get(), "ao");
                    lighting->recordCompose(worker, drawCall.lighting.sceneIndex);
                    gpuMarker(worker->commandList.get(), "compose");
                    lighting->recordSky(worker, drawCall.lighting.sceneIndex);
                    gpuMarker(worker->commandList.get(), "sky");
                    worker->commandList->setFramebuffer(fbStorage->colorWriteDepthRead.get());
                    switchToGraphicsPipeline();
                }

                break;
            };
            case InstanceDrawCall::Type::PostScene: {
                if ((lighting != nullptr) && (fbStorage->colorTarget != nullptr) && lighting->postEffects->enabled()) {
                    gpuMarker(worker->commandList.get(), "raster");
                    submitDepthAccess(worker, fbStorage, true, depthState);
                    lighting->recordPostEffects(worker, drawCall.lighting.sceneIndex);
                    gpuMarker(worker->commandList.get(), "post");
                    worker->commandList->setFramebuffer(fbStorage->colorWriteDepthRead.get());
                    switchToGraphicsPipeline();
                }

                break;
            };
            case InstanceDrawCall::Type::VertexTestZ: {
                assert(testZIndexBuffer != nullptr);
                assert(fbStorage->colorTarget != nullptr);

                const interop::RSPVertexTestZCB &testZCB = drawCall.vertexTestZ;
                switchToDepthRead();

                const bool useMSAA = (fbStorage->colorTarget->multisampling.sampleCount > 0);
                const auto &rspVertexTestZ = useMSAA ? shaderLibrary->rspVertexTestZMS : shaderLibrary->rspVertexTestZ;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::WRITE));
                worker->commandList->setPipeline(rspVertexTestZ.pipeline.get());
                worker->commandList->setComputePipelineLayout(rspVertexTestZ.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &testZCB);
                worker->commandList->setComputeDescriptorSet(vertexTestZSet->get(), 0);
                worker->commandList->setComputeDescriptorSet(descRealFbSet, 1);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::READ));

                switchToGraphicsPipeline();
                break;
            };
            default:
                // Do nothing.
                break;
            }
        }

        // Mark targets for resolve.
        if (fbStorage->colorTarget != nullptr) {
            fbStorage->colorTarget->markForResolve();
        }

        if (fbStorage->depthTarget != nullptr) {
            fbStorage->depthTarget->markForResolve();
        }
    }

    void FramebufferRenderer::updateMultisampling() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
#   if RT_ENABLED
        if (rtResources != nullptr) {
            rtResources->updateMultisampling();
        }
#   endif
    }

#if RT_ENABLED
    void FramebufferRenderer::updateRaytracingScene(RenderWorker *worker, const RaytracingScene &rtScene) {
        auto &rtParams = rtResources->rtParams;
        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;

        // Only use jitter when an upscaler is active.
        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        bool jitterActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (jitterActive) {
            const int phaseCount = upscaler->getJitterPhaseCount(rtResources->textureWidth, rtScene.screenWidth);
            rtParams.pixelJitter = HaltonJitter(frameParams.frameCount, phaseCount);
        }
        else if (rtResources->antialiasingEnabled) {
            // The temporal anti-aliasing accumulates the samples of a jittered sequence.
            const int AntialiasingPhaseCount = 8;
            rtParams.pixelJitter = HaltonJitter(frameParams.frameCount, AntialiasingPhaseCount);
        }
        else {
            rtParams.pixelJitter = { 0.0f, 0.0f };
        }

        rtParams.viewport.x = rtScene.viewport.x;
        rtParams.viewport.y = rtScene.viewport.y;
        rtParams.viewport.z = rtScene.viewport.width;
        rtParams.viewport.w = rtScene.viewport.height;

        const auto &preset = rtScene.presetScene;
        rtParams.ambientBaseColor = hlslpp::float4(preset.ambientBaseColor, 0.0f);
        rtParams.ambientNoGIColor = hlslpp::float4(preset.ambientNoGIColor, 0.0f);
        rtParams.eyeLightDiffuseColor = hlslpp::float4(preset.eyeLightDiffuseColor, 0.0f);
        rtParams.eyeLightSpecularColor = hlslpp::float4(preset.eyeLightSpecularColor, 0.0f);
        rtParams.giDiffuseStrength = preset.giDiffuseStrength;
        rtParams.giBackgroundStrength = preset.giBackgroundStrength;
        rtParams.tonemapExposure = preset.tonemapExposure;
        rtParams.tonemapWhite = preset.tonemapWhite;
        rtParams.tonemapBlack = preset.tonemapBlack;

        // Visual enhancements on top of the path tracer.
        rtParams.aoRadius = enhancementValue("RT64_RT_AO_RADIUS", 120.0f);
        rtParams.aoStrength = enhancementValue("RT64_RT_AO_STRENGTH", 0.85f);
        rtParams.volumetricStrength = enhancementValue("RT64_RT_VOLUMETRIC", 0.12f);
        rtParams.volumetricDistance = enhancementValue("RT64_RT_VOLUMETRIC_DISTANCE", 4000.0f);
        rtParams.sunDiscStrength = enhancementValue("RT64_RT_SUN_DISC", 6.0f);
        rtParams.bloomStrength = enhancementValue("RT64_RT_BLOOM", 0.18f);
        rtParams.bloomThreshold = enhancementValue("RT64_RT_BLOOM_THRESHOLD", 0.8f);
        rtParams.sharpenStrength = enhancementValue("RT64_RT_SHARPEN", 0.35f);
        rtParams.vignetteStrength = enhancementValue("RT64_RT_VIGNETTE", 0.22f);
        rtParams.saturation = enhancementValue("RT64_RT_SATURATION", 1.1f);
        rtParams.contrast = enhancementValue("RT64_RT_CONTRAST", 1.06f);
        rtParams.directHistoryLength = enhancementValue("RT64_RT_DIRECT_HISTORY", 3.0f);
        rtParams.blobShadowRemoval = enhancementValue("RT64_RT_BLOB_SHADOW_REMOVAL", 1.0f);
        rtParams.volumetricAnisotropy = enhancementValue("RT64_RT_VOLUMETRIC_ANISOTROPY", 0.25f);
        rtParams.volumetricMaxPhase = enhancementValue("RT64_RT_VOLUMETRIC_MAX_PHASE", 2.0f);
        rtParams.bumpStrength = enhancementValue("RT64_RT_BUMP", 3.5f);
        rtParams.waterReflection = enhancementValue("RT64_RT_WATER_REFLECTION", 0.15f);
        rtParams.emissiveStrength = enhancementValue("RT64_RT_EMISSIVE", 1.5f);
        rtParams.emissiveThreshold = enhancementValue("RT64_RT_EMISSIVE_THRESHOLD", 0.45f);

        // The user's choice of how strong the enhancements look.
        const float intensity = enhancementIntensity;
        rtParams.aoStrength *= intensity;
        rtParams.volumetricStrength *= intensity;
        rtParams.sunDiscStrength *= intensity;
        rtParams.bloomStrength *= intensity;
        rtParams.sharpenStrength *= intensity;
        rtParams.vignetteStrength *= intensity;
        rtParams.saturation = 1.0f + (rtParams.saturation - 1.0f) * intensity;
        rtParams.contrast = 1.0f + (rtParams.contrast - 1.0f) * intensity;
        rtParams.bumpStrength *= intensity;
        const hlslpp::float3 worldUpDir = hlslpp::normalize(rtScene.worldUp);
        rtParams.worldUp = hlslpp::float4(worldUpDir.x, worldUpDir.y, worldUpDir.z, 0.0f);

        // Orthonormal basis of the world for the sky, which is defined with Y pointing up.
        hlslpp::float3 worldRightDir = rtScene.worldRight - worldUpDir * float(hlslpp::dot(rtScene.worldRight, worldUpDir));
        if (float(hlslpp::dot(worldRightDir, worldRightDir)) < 1e-6f) {
            worldRightDir = (fabsf(float(worldUpDir.x)) < 0.9f) ? hlslpp::float3(1.0f, 0.0f, 0.0f) : hlslpp::float3(0.0f, 0.0f, 1.0f);
            worldRightDir = worldRightDir - worldUpDir * float(hlslpp::dot(worldRightDir, worldUpDir));
        }

        worldRightDir = hlslpp::normalize(worldRightDir);
        hlslpp::float3 worldForwardDir = hlslpp::cross(worldRightDir, worldUpDir);
        if (float(hlslpp::dot(worldForwardDir, rtScene.worldForward)) < 0.0f) {
            worldForwardDir = -worldForwardDir;
        }

        rtParams.worldRight = hlslpp::float4(worldRightDir.x, worldRightDir.y, worldRightDir.z, 0.0f);
        rtParams.worldForward = hlslpp::float4(worldForwardDir.x, worldForwardDir.y, worldForwardDir.z, 0.0f);
        rtParams.worldOrigin = rtScene.worldOrigin;

        // Procedural sky that replaces the sky of the game's background.
        static const auto skyStartTime = std::chrono::steady_clock::now();
        const float skySeconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - skyStartTime).count();
        rtParams.skyMode = proceduralSkyEnabled ? enhancementValue("RT64_RT_SKY", 1.0f) : 0.0f;
        rtParams.skyExposure = enhancementValue("RT64_RT_SKY_EXPOSURE", 1.0f);
        rtParams.skySaturation = enhancementValue("RT64_RT_SKY_SATURATION", 1.25f);
        rtParams.cloudCoverage = enhancementValue("RT64_RT_CLOUD_COVERAGE", 0.5f);
        rtParams.cloudDensity = enhancementValue("RT64_RT_CLOUD_DENSITY", 3.0f);
        rtParams.cloudScale = enhancementValue("RT64_RT_CLOUD_SCALE", 1.0f);
        rtParams.skyTime = fmodf(skySeconds * enhancementValue("RT64_RT_CLOUD_SPEED", 1.0f), 20000.0f);
        rtParams.skyGI = enhancementValue("RT64_RT_SKY_GI", 0.5f);
        rtParams.cloudHeight = enhancementValue("RT64_RT_CLOUD_HEIGHT", 30000.0f);
        rtParams.cloudShadows = enhancementValue("RT64_RT_CLOUD_SHADOWS", 0.5f) * intensity;
        rtParams.spriteVolume = enhancementValue("RT64_RT_SPRITE_VOLUME", 0.9f) * intensity;
        rtParams.foliageWind = enhancementValue("RT64_RT_FOLIAGE_WIND", 0.0f) * intensity;
        rtParams.foliageDetail = enhancementValue("RT64_RT_FOLIAGE_DETAIL", 0.0f) * intensity;
        rtParams.textureSmoothing = (intensity > 0.0f) ? enhancementValue("RT64_RT_TEXTURE_SMOOTHING", 1.0f) : 0.0f;
        rtParams.skyTint = hlslpp::float4(enhancementValue("RT64_RT_SKY_TINT_R", 0.8f), enhancementValue("RT64_RT_SKY_TINT_G", 0.9f), enhancementValue("RT64_RT_SKY_TINT_B", 1.15f), 0.0f);

        // The sun is the light placed much further away than anything else in the scene.
        rtParams.sunDirection = { 0.0f, 0.0f, 0.0f, 0.0f };
        rtParams.sunColor = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint32_t i = 0; i < rtScene.lightCount; i++) {
            const interop::PointLight &light = rtScene.pointLights[i];
            const float distance = sqrtf(light.position.x * light.position.x + light.position.y * light.position.y + light.position.z * light.position.z);
            if (distance > 1e6f) {
                rtParams.sunDirection = hlslpp::float4(light.position.x / distance, light.position.y / distance, light.position.z / distance, 1.0f);
                // Only the tint of the sun is used by the sky and the scattering, so they don't change with its intensity.
                const float maxComponent = std::max(std::max(light.diffuseColor.x, light.diffuseColor.y), std::max(light.diffuseColor.z, 1e-6f));
                rtParams.sunColor = hlslpp::float4(light.diffuseColor.x / maxComponent, light.diffuseColor.y / maxComponent, light.diffuseColor.z / maxComponent, 0.0f);
                break;
            }
        }

        // Colors of the procedural sky for the bounces: overhead, and at the horizon towards and away from the sun.
        if ((rtParams.sunDirection.w != 0.0f) && (rtParams.skyMode > 0.0f)) {
            const hlslpp::float3 sunRT = hlslpp::float3(rtParams.sunDirection.x, rtParams.sunDirection.y, rtParams.sunDirection.z);
            hlslpp::float3 sunSky = { float(hlslpp::dot(sunRT, worldRightDir)), float(hlslpp::dot(sunRT, worldUpDir)), float(hlslpp::dot(sunRT, worldForwardDir)) };
            sunSky = hlslpp::normalize(sunSky);
            hlslpp::float3 sunHorizontal = { float(sunSky.x), 0.0f, float(sunSky.z) };
            sunHorizontal = (float(hlslpp::length(sunHorizontal)) > 1e-4f) ? hlslpp::normalize(sunHorizontal) : hlslpp::float3(1.0f, 0.0f, 0.0f);
            const hlslpp::float3 towardsSun = hlslpp::normalize(sunHorizontal + hlslpp::float3(0.0f, 0.08f, 0.0f));
            const hlslpp::float3 awayFromSun = hlslpp::normalize(-sunHorizontal + hlslpp::float3(0.0f, 0.08f, 0.0f));
            auto skyColor = [&](hlslpp::float3 direction) {
                hlslpp::float3 color = atmosphereScattering(direction, sunSky) * rtParams.skyExposure;
                const float luma = float(color.x) * 0.2126f + float(color.y) * 0.7152f + float(color.z) * 0.0722f;
                color = hlslpp::float3(luma, luma, luma) + (color - hlslpp::float3(luma, luma, luma)) * rtParams.skySaturation;
                color = hlslpp::max(color, hlslpp::float3(0.0f, 0.0f, 0.0f)) * hlslpp::float3(float(rtParams.skyTint.x), float(rtParams.skyTint.y), float(rtParams.skyTint.z));
                return hlslpp::float4(color.x, color.y, color.z, 0.0f);
            };

            rtParams.skyZenithColor = skyColor(hlslpp::float3(0.0f, 1.0f, 0.0f));
            rtParams.skyHorizonSunColor = skyColor(towardsSun);
            rtParams.skyHorizonAwayColor = skyColor(awayFromSun);
        }

        // Without a sun (interiors) the ambient light is the only light left, so it's raised to keep the brightness
        // the textures and vertex colors were made for. Outdoors it makes up for the low sun lighting the ground at a
        // grazing angle.
        const bool sunFound = (rtParams.sunDirection.w != 0.0f);
        const float ambientScale = sunFound ? enhancementValue("RT64_RT_OUTDOOR_AMBIENT", 1.1f) : enhancementValue("RT64_RT_INDOOR_AMBIENT", 1.5f);
        rtParams.ambientBaseColor = rtParams.ambientBaseColor * hlslpp::float4(ambientScale, ambientScale, ambientScale, 0.0f);
        rtParams.ambientNoGIColor = rtParams.ambientNoGIColor * hlslpp::float4(ambientScale, ambientScale, ambientScale, 0.0f);

        const auto &proj = rtScene.curProjMatrix;
        rtParams.fovRadians = fovFromProj(proj);
        rtParams.nearDist = nearPlaneFromProj(proj);
        rtParams.farDist = farPlaneFromProj(proj);

        if (isnan(rtParams.fovRadians)) {
            rtParams.fovRadians = 0.75f;
        }

        if (isnan(rtParams.nearDist)) {
            rtParams.nearDist = 1.0f;
        }

        if (isnan(rtParams.farDist)) {
            rtParams.farDist = 1000.0f;
        }

        rtParams.view = rtScene.curViewMatrix;
        static const bool printView = (getenv("RT64_RT_PRINT_VIEW") != nullptr);
        static bool lastHadSun = false;
        if (printView && ((rtParams.sunDirection.w != 0.0f) != lastHadSun)) {
            lastHadSun = (rtParams.sunDirection.w != 0.0f);
            fprintf(stderr, "RT sun %s at frame %u\n", lastHadSun ? "on" : "off", frameParams.frameCount);
        }

        if (printView && ((frameParams.frameCount % 60) == 0)) {
            const auto &v = rtScene.curViewMatrix;
            fprintf(stderr, "RT view: [%.3f %.3f %.3f] [%.3f %.3f %.3f] [%.3f %.3f %.3f] t=[%.1f %.1f %.1f] sun=[%.3f %.3f %.3f]\n",
                v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2], v[3][0], v[3][1], v[3][2],
                rtParams.sunDirection.x, rtParams.sunDirection.y, rtParams.sunDirection.z);
            const auto &pm = rtScene.curProjMatrix;
            fprintf(stderr, "RT proj: %.3f %.3f %.3f %.3f | %.3f %.3f\n", pm[0][0], pm[1][1], pm[2][2], pm[2][3], pm[3][2], pm[3][3]);
        }
        rtParams.projection = rtScene.curProjMatrix;

        rtParams.viewI = hlslpp::inverse(rtParams.view);
        rtParams.projectionI = hlslpp::inverse(rtParams.projection);
        rtParams.viewProj = hlslpp::mul(rtParams.view, rtParams.projection);
        rtParams.prevViewProj = hlslpp::mul(rtScene.prevViewMatrix, rtScene.prevProjMatrix);

        // TODO: There's probably a way to compute this without calculating the FOV/Near/Far values.
        // Pinhole camera vectors to generate non-normalized ray direction.
        // TODO: Make a fake target and focal distance at the midpoint of the near/far planes
        // until the game sends that data in some way in the future.
        const float FocalDistance = (rtParams.nearDist + rtParams.farDist) / 2.0f;
        const hlslpp::float3 Up(0.0f, 1.0f, 0.0f);
        const hlslpp::float3 Pos = viewPositionFrom(rtParams.viewI);
        const hlslpp::float3 Target = Pos + viewDirectionFrom(rtParams.viewI) * FocalDistance;
        hlslpp::float3 cameraW = hlslpp::normalize(Target - Pos) * FocalDistance;
        hlslpp::float3 cameraU = hlslpp::normalize(hlslpp::cross(cameraW, Up));
        hlslpp::float3 cameraV = hlslpp::normalize(hlslpp::cross(cameraU, cameraW));
        const float ulen = FocalDistance * std::tan(rtParams.fovRadians * 0.5f);// * rtParams.aspectRatio;
        const float vlen = FocalDistance * std::tan(rtParams.fovRadians * 0.5f);
        cameraU = cameraU * ulen;
        cameraV = cameraV * vlen;
        rtParams.cameraU = hlslpp::float4(cameraU, 0.0f);
        rtParams.cameraV = hlslpp::float4(cameraV, 0.0f);
        rtParams.cameraW = hlslpp::float4(cameraW, 0.0f);

        // Enable light reprojection if denoising is enabled.
        rtParams.diReproject = !rtResources->skipReprojection && rtResources->denoiserEnabled && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;

        rtParams.giReproject = !rtResources->skipReprojection && rtResources->denoiserEnabled && (rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;
        rtParams.binaryLockMask = (rtResources->upscalerMode != UpscaleMode::FSR);
        rtParams.interleavedRastersCount = interleavedRastersCount;
        
        rtResources->updateTopLevelASResources(worker, instanceDrawCallVector, rtScene.instanceIndices);
        rtResources->updateLightsBuffer(worker, rtScene);
    }
    
    void FramebufferRenderer::submitRaytracingScene(RenderWorker *worker, RenderTarget *colorTarget, const RaytracingScene &rtScene) {
        // Unbind any render targets.
        worker->commandList->setFramebuffer(nullptr);

        // Resolve the color target if necessary before using it as the RT scene background.
        colorTarget->resolveTarget(worker, shaderLibrary);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(colorTarget->getResolvedTexture(), RenderTextureLayout::SHADER_READ));

        if (rtResources->transitionOutputBuffers) {
            RenderTextureBarrier afterCreationBarriers[] = {
                RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCreationBarriers, uint32_t(std::size(afterCreationBarriers)));
            rtResources->transitionOutputBuffers = false;
        }

        // Make sure all these buffers are usable as UAVs.
        RenderTextureBarrier preDispatchBarriers[] = {
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[1].get(), RenderTextureLayout::GENERAL)
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, preDispatchBarriers, uint32_t(std::size(preDispatchBarriers)));
        
        // The vertex data is read by the raytracing shaders.
        thread_local std::vector<RenderBufferBarrier> inputBarriers;
        inputBarriers.clear();
        for (RenderBuffer *buffer : rtInputBuffers) {
            if (buffer != nullptr) {
                inputBarriers.emplace_back(buffer, RenderBufferAccess::READ);
            }
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, inputBarriers);

        // Debugging aid: RT64_RT_PASSES is a mask of the ray generation passes that are dispatched
        // (1 primary, 2 direct, 4 indirect, 8 reflection, 16 refraction).
        static const uint32_t passMask = (getenv("RT64_RT_PASSES") != nullptr) ? uint32_t(strtoul(getenv("RT64_RT_PASSES"), nullptr, 0)) : 0xFFU;
        auto traceRaysPass = [&](uint32_t rayGenIndex) {
            if (passMask & (1U << rayGenIndex)) {
                rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = rayGenIndex;
                worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);
            }
        };

        // Bind pipeline and dispatch primary rays.
        RT64_LOG_PRINTF("Dispatching primary rays");
        Framebuffer &framebuffer = framebufferVector[rtFramebufferIndex];
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        worker->commandList->setPipeline(rtState->pipeline.get());
        worker->commandList->setRaytracingPipelineLayout(rtPipelineLayout);
        worker->commandList->setRaytracingDescriptorSet(descCommonSet->get(), 0);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 1);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 2);
        worker->commandList->setRaytracingDescriptorSet(descRealFbSet, 3);
        traceRaysPass(0);

        // Barriers for shading buffers before dispatching secondary rays.
        RenderTextureBarrier shadingBarriers[] = {
            RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL),
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, shadingBarriers, uint32_t(std::size(shadingBarriers)));

        // Dispatch rays for direct light.
        RT64_LOG_PRINTF("Dispatching direct light rays");
        traceRaysPass(1);

        // Dispatch rays for indirect light.
        RT64_LOG_PRINTF("Dispatching indirect light rays");
        traceRaysPass(2);

        // Wait until indirect light is done before dispatching reflection or refraction rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL));

        // Dispatch rays for refraction.
        RT64_LOG_PRINTF("Dispatching refraction rays");
        traceRaysPass(4);

        // Wait until refraction is done before dispatching reflection rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL));

        // Reflection passes.
        int reflections = rtResources->maxReflections;
        while (reflections > 0) {
            // Dispatch rays for reflection.
            RT64_LOG_PRINTF("Dispatching reflection rays");
            traceRaysPass(3);
            reflections--;

            // Add a barrier to wait for the input UAVs to be finished if there's more passes left to be done.
            if (reflections > 0) {
                RenderTextureBarrier newInputBarriers[] = {
                    RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, newInputBarriers, uint32_t(std::size(newInputBarriers)));
            }
        }

        // Restore the vertex data to the state expected by the rasterizer.
        for (RenderBufferBarrier &barrier : inputBarriers) {
            barrier.accessBits = RenderBufferAccess::READ;
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, inputBarriers);

        // Copy direct light raw buffer to the first direct filtered buffer.
        {
            RenderTexture *source = rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredDirectLightTexture[1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Copy indirect light raw buffer to the first indirect filtered buffer.
        bool denoiseGI = rtResources->denoiserEnabled && (rtResources->rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS);
        {
            RenderTexture *source = rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredIndirectLightTexture[denoiseGI ? 0 : 1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Apply a gaussian filter to the indirect light with a compute shader.
        if (denoiseGI) {
            for (int i = 0; i < 5; i++) {
                const uint32_t ThreadGroupWorkCount = 8;
                uint32_t dispatchX = (rtResources->textureWidth + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                uint32_t dispatchY = (rtResources->textureHeight + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                const ShaderRecord &gaussianFilter = shaderLibrary->gaussianFilterRGB3x3;
                interop::TextureCB textureCB;
                textureCB.TextureSize = { rtResources->textureWidth, rtResources->textureHeight };
                textureCB.TexelSize = { 1.0f / rtResources->textureWidth, 1.0f / rtResources->textureHeight };

                worker->commandList->setPipeline(gaussianFilter.pipeline.get());
                worker->commandList->setComputePipelineLayout(gaussianFilter.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &textureCB);
                worker->commandList->setComputeDescriptorSet(rtResources->indirectFilterSets[i % 2]->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);

                RenderTextureBarrier afterBlurBarriers[] = {
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 1 : 0].get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 0 : 1].get(), RenderTextureLayout::SHADER_READ)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterBlurBarriers, uint32_t(std::size(afterBlurBarriers)));
            }
        }

        // Compose the output buffer.
        RenderTexture *rtOutputCur = rtResources->outputTexture[rtResources->swapBuffers ? 1 : 0].get();

        // Barriers for shading buffers after rays are finished.
        RenderTextureBarrier afterDispatchBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterDispatchBarriers, uint32_t(std::size(afterDispatchBarriers)));

        // Set the output as the current render target.
        worker->commandList->setFramebuffer(rtResources->outputFramebuffer[rtResources->swapBuffers ? 1 : 0].get());

        // Apply the scissor and viewport to the size of the output texture.
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(rtResources->textureWidth), float(rtResources->textureHeight)));
        worker->commandList->setScissors(RenderRect(0, 0, rtResources->textureWidth, rtResources->textureHeight));

        // Draw the raytracing output.
        RT64_LOG_PRINTF("Composing the raytracing output");
        const ShaderRecord &composeShader = shaderLibrary->compose;
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->setPipeline(composeShader.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(composeShader.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->composeSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Switch resources to the correct states after composing the image
        RenderTextureBarrier afterComposeBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, afterComposeBarriers, uint32_t(std::size(afterComposeBarriers)));

        // Temporal anti-aliasing.
        Upscaler *activeUpscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        if (rtResources->antialiasingEnabled && !(rtResources->upscaleActive && (activeUpscaler != nullptr))) {
            const uint32_t pairIndex = rtResources->swapBuffers ? 1 : 0;
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
                RenderTextureBarrier(rtResources->antialiasedTexture[pairIndex].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->antialiasedTexture[pairIndex ^ 1].get(), RenderTextureLayout::SHADER_READ)
            });

            interop::TemporalAACB aaCB;
            aaCB.TextureSize = { rtResources->textureWidth, rtResources->textureHeight };
            aaCB.TexelSize = { 1.0f / rtResources->textureWidth, 1.0f / rtResources->textureHeight };
            aaCB.BlendFactor = 0.1f;
            aaCB.Reset = rtResources->skipReprojection ? 1 : 0;

            const uint32_t ThreadGroupWorkCount = 8;
            const ShaderRecord &temporalAA = shaderLibrary->temporalAA;
            worker->commandList->setPipeline(temporalAA.pipeline.get());
            worker->commandList->setComputePipelineLayout(temporalAA.pipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &aaCB);
            worker->commandList->setComputeDescriptorSet(rtResources->antialiasingSets[pairIndex]->get(), 0);
            worker->commandList->dispatch((rtResources->textureWidth + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount, (rtResources->textureHeight + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount, 1);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(rtResources->antialiasedTexture[pairIndex].get(), RenderTextureLayout::SHADER_READ));
        }

        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;
        if (lumaActive) {
            const uint32_t ThreadGroupWorkRegionDim = 8;
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

            RT64_LOG_PRINTF("Do the downscaling shader");
            {
                // Execute the compute shader for downscaling the image.
                const ShaderRecord &bicubicScaling = shaderLibrary->bicubicScaling;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::BicubicCB bicubicCB;
                bicubicCB.InputResolution = { rtResources->textureWidth, rtResources->textureHeight };
                bicubicCB.OutputResolution = { rtResources->textureWidth / 8, rtResources->textureHeight / 8 };
                worker->commandList->setPipeline(bicubicScaling.pipeline.get());
                worker->commandList->setComputePipelineLayout(bicubicScaling.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &bicubicCB);
                worker->commandList->setComputeDescriptorSet(rtResources->downscaleSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }


            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(rtResources->lumaHistogramBuffer.get(), RenderBufferAccess::READ | RenderBufferAccess::WRITE), RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));

            RT64_LOG_PRINTF("Do the luminance histogram shader");
            {
                // Execute the compute shader for the luminance histogram.
                const ShaderRecord &luminanceHistogram = shaderLibrary->luminanceHistogram;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::LuminanceHistogramCB histogramCB;
                histogramCB.inputWidth = rtResources->textureWidth / 8;
                histogramCB.inputHeight = rtResources->textureHeight / 8;
                histogramCB.minLuminance = rtScene.presetScene.minLuminance;
                histogramCB.oneOverLuminanceRange = 1.0f / rtScene.presetScene.luminanceRange;
                worker->commandList->setPipeline(luminanceHistogram.pipeline.get());
                worker->commandList->setComputePipelineLayout(luminanceHistogram.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &histogramCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(rtResources->lumaHistogramBuffer.get(), RenderBufferAccess::READ));

            RT64_LOG_PRINTF("Do the luminance average shader");
            {
                // Execute the compute shader for the luminance histogram average.
                const ShaderRecord &histogramAverage = shaderLibrary->histogramAverage;
                interop::HistogramAverageCB averageCB;
                averageCB.pixelCount = (rtResources->textureWidth / 8) * (rtResources->textureHeight / 8);
                averageCB.minLuminance = rtScene.presetScene.minLuminance;
                averageCB.luminanceRange = rtScene.presetScene.luminanceRange;
                averageCB.timeDelta = rtScene.deltaTime;
                averageCB.tau = rtScene.presetScene.lumaUpdateTime;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramAverage.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramAverage.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &averageCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaAvgSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }

            RT64_LOG_PRINTF("Do the histogram clear shader");
            {
                // Execute the compute shader for clearing the luminance histogram.
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(rtResources->lumaHistogramBuffer.get(), RenderBufferAccess::READ | RenderBufferAccess::WRITE));
                const ShaderRecord &histogramClear = shaderLibrary->histogramClear;
                worker->commandList->setPipeline(histogramClear.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramClear.pipelineLayout.get());
                worker->commandList->setComputeDescriptorSet(rtResources->lumaClearSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
            }
        }
        else {
            RT64_LOG_PRINTF("Do the histogram set shader");
            {
                // Execute the compute shader for setting the luminance value.
                const ShaderRecord &histogramSet = shaderLibrary->histogramSet;
                interop::HistogramSetCB setCB;
                setCB.luminanceValue = rtScene.presetScene.minLuminance;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramSet.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramSet.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &setCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSetSet->get(), 0);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }
        }

        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        const bool upscalerActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (upscalerActive) {
            thread_local std::vector<RenderTextureBarrier> beforeBarriers;
            thread_local std::vector<RenderTextureBarrier> afterBarriers;
            beforeBarriers.clear();
            afterBarriers.clear();

            beforeBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::GENERAL));
            afterBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));
            RenderTexture *rtDepthCur = rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get();
            if (upscaler->requiresNonShaderResourceInputs()) {
                for (RenderTexture *res : { rtOutputCur, rtResources->flowTexture.get(), rtResources->reactiveMaskTexture.get(), rtResources->lockMaskTexture.get(), rtDepthCur }) {
                    beforeBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                    afterBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                }
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, beforeBarriers);

            Upscaler::UpscaleParameters params;
            params.inRect = { 0, 0, static_cast<int>(rtResources->textureWidth), static_cast<int>(rtResources->textureHeight) };
            params.inDiffuseAlbedo = rtResources->diffuseTexture.get();
            params.inSpecularAlbedo = rtResources->shadingSpecularTexture.get();
            params.inNormalRoughness = rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get();
            params.inColor = rtOutputCur;
            params.inFlow = rtResources->flowTexture.get();
            params.inReactiveMask = rtResources->upscalerReactiveMask ? rtResources->reactiveMaskTexture.get() : nullptr;
            params.inLockMask = rtResources->upscalerLockMask ? rtResources->lockMaskTexture.get() : nullptr;
            params.inDepth = rtDepthCur;
            params.outColor = rtResources->upscaledOutputTexture.get();
            params.jitterX = -rtResources->rtParams.pixelJitter.x;
            params.jitterY = -rtResources->rtParams.pixelJitter.y;
            params.deltaTime = rtScene.deltaTime * 1000.0f;
            params.nearPlane = rtResources->rtParams.nearDist;
            params.farPlane = rtResources->rtParams.farDist;
            params.fovY = rtResources->rtParams.fovRadians;
            params.resetAccumulation = false; // TODO: Make this configurable via the API.
            upscaler->upscale(worker, params);

            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterBarriers);
        }

        // Bloom of the bright parts of the image.
        RenderTexture *bloom0 = rtResources->bloomTexture[0].get();
        RenderTexture *bloom1 = rtResources->bloomTexture[1].get();
        if (rtResources->rtParams.bloomStrength > 0.0f) {
            RenderTexture *bloomInput = rtOutputCur;
            uint32_t bloomInputWidth = rtResources->textureWidth;
            uint32_t bloomInputHeight = rtResources->textureHeight;
            if (upscalerActive) {
                bloomInput = rtResources->upscaledOutputTexture.get();
                bloomInputWidth = rtResources->screenWidth;
                bloomInputHeight = rtResources->screenHeight;
            }
            else if (rtResources->antialiasingEnabled) {
                bloomInput = rtResources->antialiasedTexture[rtResources->swapBuffers ? 1 : 0].get();
            }

            const ShaderRecord &bloom = shaderLibrary->bloom;
            const uint32_t ThreadGroupWorkCount = 8;
            const uint32_t dispatchX = (rtResources->bloomWidth + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
            const uint32_t dispatchY = (rtResources->bloomHeight + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
            interop::BloomCB bloomCB;
            bloomCB.OutputSize = { rtResources->bloomWidth, rtResources->bloomHeight };
            bloomCB.Threshold = rtResources->rtParams.bloomThreshold;
            worker->commandList->setPipeline(bloom.pipeline.get());
            worker->commandList->setComputePipelineLayout(bloom.pipelineLayout.get());

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, { RenderTextureBarrier(bloomInput, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(bloom0, RenderTextureLayout::GENERAL) });
            bloomCB.InputTexelSize = { 1.0f / bloomInputWidth, 1.0f / bloomInputHeight };
            bloomCB.Mode = 0;
            worker->commandList->setComputePushConstants(0, &bloomCB);
            worker->commandList->setComputeDescriptorSet(rtResources->bloomSets[0]->get(), 0);
            worker->commandList->dispatch(dispatchX, dispatchY, 1);

            bloomCB.InputTexelSize = { 1.0f / rtResources->bloomWidth, 1.0f / rtResources->bloomHeight };
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, { RenderTextureBarrier(bloom0, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(bloom1, RenderTextureLayout::GENERAL) });
            bloomCB.Mode = 1;
            worker->commandList->setComputePushConstants(0, &bloomCB);
            worker->commandList->setComputeDescriptorSet(rtResources->bloomSets[1]->get(), 0);
            worker->commandList->dispatch(dispatchX, dispatchY, 1);

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, { RenderTextureBarrier(bloom1, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(bloom0, RenderTextureLayout::GENERAL) });
            bloomCB.Mode = 2;
            worker->commandList->setComputePushConstants(0, &bloomCB);
            worker->commandList->setComputeDescriptorSet(rtResources->bloomSets[2]->get(), 0);
            worker->commandList->dispatch(dispatchX, dispatchY, 1);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(bloom0, RenderTextureLayout::SHADER_READ));

        // Set the final render target. Apply the same scissor and viewport that was determined for the raytracing step.
        worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        worker->commandList->setViewports(rtScene.viewport);
        worker->commandList->setScissors(rtScene.scissor);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);

        // Draw final output.
        const ShaderRecord &postProcess = shaderLibrary->postProcess;
        worker->commandList->setPipeline(postProcess.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(postProcess.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->postProcessSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Draw debug view on top.
        if (rtResources->rtParams.visualizationMode != interop::VisualizationMode::Final) {
            const ShaderRecord &debugShader = shaderLibrary->debug;
            worker->commandList->setPipeline(debugShader.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(debugShader.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }

        // Mark targets for resolve.
        colorTarget->markForResolve();
    }
#endif

    void FramebufferRenderer::submitRSPSmoothNormalCompute(RenderWorker *worker, const OutputBuffers *outputBuffers) {
        if (rspSmoothNormalVector.empty()) {
            return;
        }
        
        if (smoothNormalGroups.empty() || (smoothNormalGroupsBuffer.get() == nullptr)) {
            return;
        }

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(outputBuffers->worldNormBuffer.buffer.get(), RenderBufferAccess::WRITE));

        // All the ranges in one dispatch, with one group per tile of triangles of each range (see endFramebuffers).
        const auto &rspSmoothNormal = shaderLibrary->rspSmoothNormal;
        RSPSmoothNormalGenerationCB cb;
        cb.indexStart = uint32_t(smoothNormalGroups.size());
        cb.indexCount = 0;
        cb.creaseCosine = 0.0f;
        worker->commandList->setPipeline(rspSmoothNormal.pipeline.get());
        worker->commandList->setComputePipelineLayout(rspSmoothNormal.pipelineLayout.get());
        worker->commandList->setComputePushConstants(0, &cb);
        worker->commandList->setComputeDescriptorSet(smoothDescSet->get(), 0);
        worker->commandList->dispatch(uint32_t(smoothNormalGroups.size()), 1, 1);

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(outputBuffers->worldNormBuffer.buffer.get(), RenderBufferAccess::READ));
    }

    void FramebufferRenderer::recordSetup(RenderWorker *worker, std::vector<BufferUploader *> bufferUploaders, RSPProcessor *rspProcessor,
        VertexProcessor *vertexProcessor, const OutputBuffers *outputBuffers, bool rtEnabled) {
        if (!dummyColorTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyColorTarget.get(), RenderTextureLayout::SHADER_READ));
            dummyColorTargetTransitioned = true;
        }

        if (!dummyDepthTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ));
            dummyDepthTargetTransitioned = true;
        }

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListBeforeBarriers(worker);
        }

        shaderUploader->commandListBeforeBarriers(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListCopyResources(worker);
        }

        shaderUploader->commandListCopyResources(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListAfterBarriers(worker);
        }

        gpuMarker(worker->commandList.get(), "uploads");
        if (rspProcessor != nullptr) {
            rspProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
            gpuMarker(worker->commandList.get(), "rsp");
        }

        if (vertexProcessor != nullptr) {
            vertexProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
            gpuMarker(worker->commandList.get(), "world vertices");
        }

        if (lightingBuffersActive && !rtEnabled && (vertexProcessor != nullptr) && !rspSmoothNormalVector.empty()) {
            submitRSPSmoothNormalCompute(worker, outputBuffers);
            gpuMarker(worker->commandList.get(), "smooth normals");
        }

#   if RT_ENABLED
        if (rtEnabled) {
            assert(rtResources != nullptr);
            if (!rtResources->bottomLevelASVector.empty()) {
                if (!rspSmoothNormalVector.empty()) {
                    submitRSPSmoothNormalCompute(worker, outputBuffers);
                }

                rtResources->submitBottomLevelASCreation(worker);
                rtResources->submitTopLevelASCreation(worker);
            }
        }
#   endif

        shaderUploader->commandListAfterBarriers(worker);
        pendingUploaders = bufferUploaders;

        if (!dynamicTextureBarrierVector.empty()) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, dynamicTextureBarrierVector);
        }
    }

    void FramebufferRenderer::recordFramebuffer(RenderWorker *worker, uint32_t framebufferIndex) {
        // Submit all transition barriers first.
        thread_local std::vector<RenderTextureBarrier> startBarriers;
        startBarriers.clear();

        const Framebuffer &framebuffer = framebufferVector[framebufferIndex];
        for (RenderTarget *target : framebuffer.transitionRenderTargetSet) {
            startBarriers.emplace_back(RenderTextureBarrier(target->getResolvedTexture(), RenderTextureLayout::SHADER_READ));
        }
        
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        RenderTarget *colorTarget = targetDrawCall.fbStorage->colorTarget;
        RenderTarget *depthTarget = targetDrawCall.fbStorage->depthTarget;
        if (colorTarget != nullptr) {
            startBarriers.emplace_back(RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE));
        }

        startBarriers.emplace_back(RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, startBarriers);

        // The shadow map of the enhanced lighting is drawn before the framebuffer that uses it.
        gpuMarker(worker->commandList.get(), "framebuffer setup");
        if ((lighting != nullptr) && framebuffer.hasLighting) {
            if ((worldPosBuffer != nullptr) && (worldNormBuffer != nullptr)) {
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, { RenderBufferBarrier(worldPosBuffer, RenderBufferAccess::READ), RenderBufferBarrier(worldNormBuffer, RenderBufferAccess::READ) });
            }

            lighting->recordShadowMap(worker, descCommonSet->get(), descTextureSet->get(), framebuffer.descDummyFbSet->get(), shadowVertexViews.data(), vertexInputSlots.data(),
                uint32_t(shadowVertexViews.size()), &indexBufferView, instanceDrawCallVector);
            gpuMarker(worker->commandList.get(), "shadow map");
        }

        bool depthState = false;
        worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorDepthWrite.get());
        for (const auto &pair : targetDrawCall.sceneIndices) {
#       if RT_ENABLED
            if (pair.second) {
                const auto &rtScene = targetDrawCall.rtScenes[pair.first];

                // Draw all the interleaved rasterized buffers that will be used in the render target.
                thread_local std::vector<RenderTextureBarrier> interleavedBarriers;
                interleavedBarriers.clear();
                for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                    bool interleavedDepthState = false;
                    const uint32_t sceneIndex = rtScene.interleavedRasters[i].rasterSceneIndex;
                    RenderTarget *colorRenderTarget = rtResources->interleavedColorTargetVector[i].get();
                    RenderTarget *depthRenderTarget = rtResources->interleavedDepthTargetVector[i].get();
                    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                        RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
                        RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE)
                    });

                    RenderFramebufferStorage *fbStorage = rtResources->interleavedFramebufferStorageVector[i].get();
                    worker->commandList->setFramebuffer(fbStorage->colorDepthWrite.get());
                    worker->commandList->clearColor();
                    worker->commandList->clearDepth();

                    submitDepthAccess(worker, fbStorage, false, interleavedDepthState);
                    submitRasterScene(worker, framebuffer, fbStorage, targetDrawCall.rasterScenes[sceneIndex], interleavedDepthState);

                    // Resolve the interleaved scene.
                    // TODO: Depth textures need to be thrown into a separate view vector for multisampled textures.
                    fbStorage->colorTarget->resolveTarget(worker, shaderLibrary);

                    interleavedBarriers.emplace_back(RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::SHADER_READ));
                    interleavedBarriers.emplace_back(RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_READ));
                }

                if (!interleavedBarriers.empty()) {
                    worker->commandList->barriers(RenderBarrierStage::COMPUTE, interleavedBarriers);
                }

                submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
                submitRaytracingScene(worker, targetDrawCall.fbStorage->colorTarget, rtScene);
            }
            else
#       endif
            {
                const RasterScene &rasterScene = targetDrawCall.rasterScenes[pair.first];
                submitDepthAccess(worker, targetDrawCall.fbStorage, false, depthState);
                submitRasterScene(worker, framebuffer, targetDrawCall.fbStorage, rasterScene, depthState);
            }
        }
    }

    void FramebufferRenderer::waitForUploaders() {
        shaderUploader->wait();

        for (BufferUploader *uploader : pendingUploaders) {
            uploader->wait();
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged) {
        assert(rtResources != nullptr);
        rtResources->setRaytracingConfig(rtConfig, resolutionChanged);
    }
#endif

    void FramebufferRenderer::addFramebuffer(const DrawParams &p) {
        assert(p.fbStorage != nullptr);
        
        // Setup framebuffer pair data and descriptor set.
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        interop::FramebufferParams fbParams;
        fbParams.resolution = { p.targetWidth / p.resolutionScale.x, p.targetHeight / p.resolutionScale.y };
        fbParams.resolutionScale = p.resolutionScale;
        fbParams.horizontalMisalignment = p.horizontalMisalignment;
        framebufferCount++;

        while (framebufferCount > framebufferVector.size()) {
            framebufferVector.emplace_back();
            framebufferVector.back().paramsBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(256, RenderBufferFlag::CONSTANT));
            framebufferVector.back().descRealFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
            framebufferVector.back().descDummyFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
        }

        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        void *paramBufferBytes = framebuffer.paramsBuffer->map();
        memcpy(paramBufferBytes, &fbParams, sizeof(interop::FramebufferParams));
        framebuffer.paramsBuffer->unmap();

        RenderTexture *backgroundColorTexture = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTexture() : dummyColorTarget.get();
        RenderTextureView *backgroundColorTextureView = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTextureView() : dummyColorTargetView.get();
        framebuffer.descRealFbSet->setBuffer(framebuffer.descRealFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        framebuffer.descDummyFbSet->setBuffer(framebuffer.descDummyFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundDepth, dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ, dummyDepthTargetView.get());

        // Store ubershader and other effect pipelines.
        const RasterShaderUber *rasterShaderUber = p.rasterShaderCache->getGPUShaderUber();
        rendererPipelineLayout = rasterShaderUber->pipelineLayout.get();
        postBlendDitherNoiseAddPipeline = rasterShaderUber->postBlendDitherNoiseAddPipeline.get();
        postBlendDitherNoiseSubPipeline = rasterShaderUber->postBlendDitherNoiseSubPipeline.get();
        postBlendDitherNoiseSubNegativePipeline = rasterShaderUber->postBlendDitherNoiseSubNegativePipeline.get();

        // Make a new render target draw call.
        RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        const DrawData &drawData = p.curWorkload->drawData;
        const DrawBuffers &drawBuffers = p.curWorkload->drawBuffers;
        const OutputBuffers &outputBuffers = p.curWorkload->outputBuffers;
        const RenderBuffer *screenPosRes = outputBuffers.screenPosBuffer.buffer.get();
        const RenderBuffer *tcRes = outputBuffers.genTexCoordBuffer.buffer.get();
        const RenderBuffer *indexRes = drawBuffers.faceIndicesBuffer.get();
        const RenderBuffer *shadedColRes = outputBuffers.shadedColBuffer.buffer.get();
        const RenderBuffer *worldPosRes = outputBuffers.worldPosBuffer.buffer.get();
        const RenderBuffer *worldNormRes = outputBuffers.worldNormBuffer.buffer.get();
        const RenderBuffer *worldVelRes = outputBuffers.worldVelBuffer.buffer.get();
        const RenderBuffer *triPosRes = drawBuffers.triPosBuffer.get();
        const RenderBuffer *triTcRes = drawBuffers.triTcBuffer.get();
        const RenderBuffer *triColRes = drawBuffers.triColorBuffer.get();
        const uint32_t PosStride = sizeof(float) * 4;
        const uint32_t ColStride = sizeof(float) * 4;
        const uint32_t WorldNormStride = sizeof(float) * 4;
        const uint32_t WorldVelStride = sizeof(float) * 4;
        const uint32_t TcStride = sizeof(float) * 2;
        const uint32_t IndexStride = sizeof(uint32_t);
        const uint32_t vertexCount = drawData.vertexCount();
        const uint32_t indexCount = uint32_t(drawData.faceIndices.size());
        const uint32_t rawTriVertexCount = drawData.rawTriVertexCount();
        vertexInputSlots[0] = RenderInputSlot(0, PosStride);
        vertexInputSlots[1] = RenderInputSlot(1, TcStride);
        vertexInputSlots[2] = RenderInputSlot(2, ColStride);
        indexedVertexViews[0] = RenderVertexBufferView(RenderBufferReference(screenPosRes), PosStride * vertexCount);
        indexedVertexViews[1] = RenderVertexBufferView(RenderBufferReference(tcRes), TcStride * vertexCount);
        indexedVertexViews[2] = RenderVertexBufferView(RenderBufferReference(shadedColRes), ColStride * vertexCount);
        indexBufferView = RenderIndexBufferView(RenderBufferReference(indexRes), IndexStride * indexCount, RenderFormat::R32_UINT);
        rawVertexViews[0] = RenderVertexBufferView(RenderBufferReference(triPosRes), PosStride * rawTriVertexCount);
        rawVertexViews[1] = RenderVertexBufferView(RenderBufferReference(triTcRes), TcStride * rawTriVertexCount);
        rawVertexViews[2] = RenderVertexBufferView(RenderBufferReference(triColRes), ColStride * rawTriVertexCount);
        testZIndexBuffer = outputBuffers.testZIndexBuffer.buffer.get();
        testZIndexBufferView = RenderIndexBufferView(testZIndexBuffer, uint32_t(outputBuffers.testZIndexBuffer.allocatedSize), RenderFormat::R32_UINT);
        shadowVertexViews[0] = RenderVertexBufferView(RenderBufferReference(worldPosRes), PosStride * vertexCount);
        shadowVertexViews[1] = indexedVertexViews[1];
        shadowVertexViews[2] = indexedVertexViews[2];
        worldPosBuffer = outputBuffers.worldPosBuffer.buffer.get();
        worldNormBuffer = outputBuffers.worldNormBuffer.buffer.get();
        framebuffer.hasLighting = false;

        RasterScene rasterScene;
        auto checkRasterScene = [&](RasterScene &rasterScene) {
            if (!rasterScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rasterScenes.size());
                targetDrawCall.rasterScenes.push_back(rasterScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, false });
                rasterScene.instanceIndices.clear();

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rasterScenes.clear();

#   if RT_ENABLED
        RaytracingScene rtScene;
        auto checkRtScene = [&](RaytracingScene &rtScene) {
            if (!rtScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rtScenes.size());
                targetDrawCall.rtScenes.push_back(rtScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, true });
                rtScene.instanceIndices.clear();

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rtScenes.clear();
#   endif

        targetDrawCall.fbStorage = p.fbStorage;
        targetDrawCall.sceneIndices.clear();

        const float SimilarityPercentage = 0.1f; // TODO: Make more strict once VI ratios are in.
        const float scissorRatio = static_cast<float>(fbPair.scissorRect.width(false, true)) / static_cast<float>(fbPair.scissorRect.height(false, true));
        const bool adjustRatio = (abs((scissorRatio / p.aspectRatioSource) - 1.0f) < SimilarityPercentage);
        const float aspectRatioScale = adjustRatio ? (p.aspectRatioTarget / p.aspectRatioSource) : 1.0f;
        InstanceDrawCall instanceDrawCall;
        interop::RenderIndices renderIndices;
        uint32_t globalCallIndex = 0;
        const float wideWidth = p.fbWidth * p.resolutionScale.x;
        const float originalWidth = p.fbWidth * p.resolutionScale.y;
        const float commonHeight = float(p.targetHeight);
        framebuffer.viewport = RenderViewport(0.0f, 0.0f, wideWidth, commonHeight);
        
        const interop::float2 halfViewportSize = { framebuffer.viewport.width / 2.0f, framebuffer.viewport.height / 2.0f };
        const interop::float2 halfPixelOffset = { 1.0f / framebuffer.viewport.width, -1.0f / framebuffer.viewport.height };
        const float middleViewport = (wideWidth / 2.0f) - (originalWidth / 2.0f);
        const float extOriginPercentage = p.extAspectPercentage;
        uint32_t vertexTestZFaceIndicesStart = 0;
        int32_t vertexTestZCallIndex = -1;
        RenderViewport viewportClip;

        // Enhanced lighting: the opaque draw calls of each 3D projection are drawn first, then the lighting is composed
        // over them and the translucent ones are drawn afterwards in their original order, so they aren't darkened.
        const bool lightingActive = p.lightingEnabled && (lighting != nullptr) && (p.fbStorage->colorTarget != nullptr) && (p.fbStorage->depthTarget != nullptr) && fbPair.depthWrite;
        int32_t lightingSceneIndex = -1;
        interop::float4x4 lightingViewProj;
        const float lightingSmoothNormalAngle = lightingActive ? enhancementValue("RT64_LIGHT_SMOOTH_NORMALS", 75.0f) : 0.0f;
        static const uint32_t SmoothNormalTriangles[] = { 0, 256, 1024, 1024 };
        const uint32_t lightingSmoothNormalTriangles = uint32_t(enhancementValue("RT64_LIGHT_SMOOTH_NORMALS_MAX", float(SmoothNormalTriangles[getRasterLightingQuality()])));
        thread_local std::vector<uint32_t> lightingDeferred;
        thread_local std::vector<interop::float4x4> lightingLitViewProjs;
        lightingDeferred.clear();
        lightingLitViewProjs.clear();
        auto closeLightingScene = [&]() {
            if (lightingSceneIndex < 0) {
                return;
            }

            InstanceDrawCall markerDrawCall;
            markerDrawCall.type = InstanceDrawCall::Type::Lighting;
            markerDrawCall.lighting.sceneIndex = uint32_t(lightingSceneIndex);
            interop::RenderIndices markerIndices = {};
            renderIndicesVector.push_back(markerIndices);
            rasterScene.instanceIndices.push_back(uint32_t(instanceDrawCallVector.size()));
            instanceDrawCallVector.push_back(markerDrawCall);
            rasterScene.instanceIndices.insert(rasterScene.instanceIndices.end(), lightingDeferred.begin(), lightingDeferred.end());
            lightingDeferred.clear();

            // The post effects of the scene go after its translucent surfaces and before anything else drawn on top.
            InstanceDrawCall postDrawCall;
            postDrawCall.type = InstanceDrawCall::Type::PostScene;
            postDrawCall.lighting.sceneIndex = uint32_t(lightingSceneIndex);
            renderIndicesVector.push_back(markerIndices);
            rasterScene.instanceIndices.push_back(uint32_t(instanceDrawCallVector.size()));
            instanceDrawCallVector.push_back(postDrawCall);
            lightingLitViewProjs.push_back(lightingViewProj);
            lightingSceneIndex = -1;
            framebuffer.hasLighting = true;
        };

        for (uint32_t pr = 0; (pr < fbPair.projectionCount) && (globalCallIndex < p.maxGameCall); pr++) {
            const Projection &proj = fbPair.projections[pr];
            if (proj.scissorRect.isNull()) {
                continue;
            }

#       if RT_ENABLED
            // TODO: Move heuristics of RT proj elsewhere?
            // TODO: Use detected scenes logic instead.
            const bool perspProj = (proj.type == Projection::Type::Perspective);
            bool rtProj = p.rtEnabled && perspProj && fbPair.depthWrite && targetDrawCall.rtScenes.empty(); // TODO: Remove the last condition once multiple heaps per RT scene are supported.

            // Make sure the matrices are compatible if we're switching to a new projection.
            bool rtProjCompatible = true;
            if (rtProj && (!rtScene.instanceIndices.empty())) {
                const float Threshold = 1e-6f;
                const float viewMatrixDiff = matrixDifference(drawData.modViewTransforms[proj.transformsIndex], rtScene.curViewMatrix);
                const float projMatrixDiff = matrixDifference(drawData.modProjTransforms[proj.transformsIndex], rtScene.curProjMatrix);
                rtProjCompatible = (viewMatrixDiff < Threshold) && (projMatrixDiff < Threshold);
            }

            // TODO: Remove this condition once multiple heaps per RT scene are supported.
            if (rtProj && !rtProjCompatible) {
                rtProj = false;
            }
#       endif
            
            auto &triangles = instanceDrawCall.triangles;
            triangles.screenScale = { 1.0f, 1.0f };
            triangles.screenOffset = halfPixelOffset;

            float projInvRatioScale = 1.0f / aspectRatioScale;
            const int16_t *viewportClipRatios = &drawData.viewportClipRatios[proj.transformsIndex * 4];
            const uint16_t viewportOrigin = drawData.viewportOrigins[proj.transformsIndex];
            if (proj.usesViewport()) {
                // The call's scissor spans the whole width of the framebuffer pair scissor. Custom origin must not be in use to be able to use the stretched viewport.
                const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                FixedRect intersectionRect = proj.scissorRect.intersection(viewport.rect(viewportClipRatios));
                bool coversWholeWidth = !intersectionRect.isEmpty() && (intersectionRect.ulx <= fbPair.scissorRect.ulx) && (intersectionRect.lrx >= fbPair.scissorRect.lrx);
                bool horizontalRatio = !intersectionRect.isEmpty() && (intersectionRect.width(true, true) > intersectionRect.height(true, true));
                bool useWideViewport = (viewportOrigin == G_EX_ORIGIN_NONE) && coversWholeWidth && horizontalRatio;
                if (useWideViewport) {
                    projInvRatioScale = 1.0f;
                }
                else {
                    triangles.screenScale.x = originalWidth / wideWidth;

                    if (viewportOrigin < G_EX_ORIGIN_NONE) {
                        const float centerOffset = ((middleViewport * viewportOrigin) / G_EX_ORIGIN_CENTER) * extOriginPercentage + middleViewport * (1.0f - extOriginPercentage);
                        triangles.screenOffset.x = halfPixelOffset.x + ((centerOffset - middleViewport) / halfViewportSize.x);
                    }
                }

                viewportClip = convertViewportRect(viewport.rect(viewportClipRatios), p.resolutionScale, p.fbWidth, projInvRatioScale, extOriginPercentage, 0.0f, viewportOrigin, viewportOrigin);
            }

            if (lightingActive) {
                const bool lightingProj = (proj.type == Projection::Type::Perspective) && proj.usesViewport();
                const interop::float4x4 &projViewProj = p.modTransformsValid ? drawData.modViewProjTransforms[proj.transformsIndex] : drawData.viewProjTransforms[proj.transformsIndex];
                if ((lightingSceneIndex >= 0) && (!lightingProj || (matrixDifference(projViewProj, lightingViewProj) > 1e-6f))) {
                    closeLightingScene();
                }

                // A projection that was already lit in this framebuffer isn't lit again, as its pixels would be lit twice.
                bool alreadyLit = false;
                for (const interop::float4x4 &litViewProj : lightingLitViewProjs) {
                    alreadyLit = alreadyLit || (matrixDifference(projViewProj, litViewProj) <= 1e-6f);
                }

                const interop::RSPViewport &rspViewport = drawData.rspViewports[proj.transformsIndex];
                const bool validViewport = (fabsf(rspViewport.scale.x) > 1e-6f) && (fabsf(rspViewport.scale.y) > 1e-6f) && (fabsf(rspViewport.scale.z) > 1e-8f);
                if (lightingProj && (lightingSceneIndex < 0) && !alreadyLit && validViewport) {
                    // Inverse of the transformations done by the RSP and the raster vertex shader, from the pixels of the
                    // framebuffer to the clip space of the projection.
                    const float resolutionX = float(p.targetWidth) / p.resolutionScale.x;
                    const float resolutionY = float(p.targetHeight) / p.resolutionScale.y;
                    const interop::float2 screenScale = triangles.screenScale;
                    const interop::float2 screenOffset = triangles.screenOffset;
                    auto pixelToClipX = [&](float pixelX) {
                        const float ndc = (pixelX / framebuffer.viewport.width) * 2.0f - 1.0f;
                        const float screen = ((ndc - screenOffset.x) / screenScale.x) * (resolutionX * 0.5f) + resolutionX * 0.5f;
                        return (screen - rspViewport.translate.x) / rspViewport.scale.x;
                    };

                    auto pixelToClipY = [&](float pixelY) {
                        const float ndc = 1.0f - (pixelY / framebuffer.viewport.height) * 2.0f;
                        const float screen = ((ndc - screenOffset.y) / screenScale.y) * (resolutionY * -0.5f) + resolutionY * 0.5f;
                        return -(screen - rspViewport.translate.y) / rspViewport.scale.y;
                    };

                    LightingSceneDesc sceneDesc;
                    sceneDesc.viewProj = projViewProj;
                    sceneDesc.view = p.modTransformsValid ? drawData.modViewTransforms[proj.transformsIndex] : drawData.viewTransforms[proj.transformsIndex];
                    sceneDesc.pixelToClip = hlslpp::float4(pixelToClipX(1.0f) - pixelToClipX(0.0f), pixelToClipY(1.0f) - pixelToClipY(0.0f), pixelToClipX(0.0f), pixelToClipY(0.0f));
                    const float maxDepth = std::min(rspViewport.translate.z + rspViewport.scale.z, 1.0f);
                    sceneDesc.depthToClip = hlslpp::float4(1.0f / rspViewport.scale.z, -rspViewport.translate.z / rspViewport.scale.z, maxDepth * enhancementValue("RT64_LIGHT_BACKGROUND_DEPTH", 0.99995f), 0.0f);
                    sceneDesc.worldRight = p.curWorkload->worldRight;
                    sceneDesc.worldUp = p.curWorkload->worldUp;
                    sceneDesc.worldForward = p.curWorkload->worldForward;
                    sceneDesc.worldOrigin = p.curWorkload->worldOrigin;
                    sceneDesc.skyHidden = p.curWorkload->skyBackgroundHint;
                    sceneDesc.lights = (proj.pointLightCount > 0) ? proj.pointLights.data() : nullptr;
                    sceneDesc.lightCount = proj.pointLightCount;
                    sceneDesc.screenScale = screenScale;
                    sceneDesc.screenOffset = screenOffset;
                    sceneDesc.viewport = framebuffer.viewport;

                    // The fog of the game, taken from the first draw call that uses it.
                    for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                        const GameCall &fogCall = proj.gameCalls[d];
                        const uint32_t faceStart = fogCall.meshDesc.faceIndicesStart;
                        if ((fogCall.callDesc.triangleCount == 0) || (faceStart >= drawData.faceIndices.size())) {
                            continue;
                        }

                        const uint32_t vertexIndex = drawData.faceIndices[faceStart];
                        if (vertexIndex >= drawData.fogIndices.size()) {
                            continue;
                        }

                        const uint32_t fogIndex = drawData.fogIndices[vertexIndex];
                        if ((fogIndex > 0) && ((fogIndex - 1) < drawData.rspFog.size())) {
                            sceneDesc.fogEnabled = true;
                            sceneDesc.fogMul = drawData.rspFog[fogIndex - 1].mul;
                            sceneDesc.fogOffset = drawData.rspFog[fogIndex - 1].offset;
                            break;
                        }
                    }

                    lightingSceneIndex = int32_t(lighting->addScene(sceneDesc, p.fbStorage->colorTarget, p.fbStorage->depthTarget));
                    lightingViewProj = projViewProj;
                }
            }

            for (uint32_t d = 0; (d < proj.gameCallCount) && (globalCallIndex < p.maxGameCall); d++) {
                const GameCall &call = proj.gameCalls[d];
                bool lightingDeferredCall = false;
                renderIndices.instanceIndex = call.callDesc.callIndex;
                renderIndices.faceIndicesStart = call.meshDesc.faceIndicesStart;
                renderIndices.rdpTileIndex = call.callDesc.tileIndex;
                renderIndices.rdpTileCount = call.callDesc.tileCount;
                renderIndices.highlightColor = call.debuggerDesc.highlightColor;
                renderIndicesVector.push_back(renderIndices);

                uint32_t cycleType = call.callDesc.otherMode.cycleType();
                if (cycleType == G_CYC_FILL) {
                    instanceDrawCall.type = InstanceDrawCall::Type::FillRect;

                    auto &clearRect = instanceDrawCall.clearRect;
                    if (call.debuggerDesc.highlightColor > 0) {
                        clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.debuggerDesc.highlightColor));
                    }
                    else {
                        if (p.fbStorage->colorTarget == nullptr) {
                            clearRect.depth = ColorConverter::D16::toF(call.callDesc.fillColor & 0xFFFF);
                        }
                        else if (fbPair.colorImage.siz == G_IM_SIZ_32b) {
                            clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.callDesc.fillColor));
                        }
                        else {
                            clearRect.color = toRenderColor(ColorConverter::RGBA16::toRGBAF(call.callDesc.fillColor & 0xFFFF));
                        }
                    }

                    float invRatioScale = 1.0f / aspectRatioScale;
                    int32_t horizontalMisalignment = 0;

                    // A rect that spans the whole width of the scissor.
                    if ((call.callDesc.rect.ulx <= fbPair.scissorRect.ulx) && (call.callDesc.rect.lrx >= fbPair.scissorRect.lrx)) {
                        invRatioScale = 1.0f;
                    }
                    // A regular rectangle that should correct its misalignment.
                    else {
                        horizontalMisalignment = int32_t(p.horizontalMisalignment);
                    }

                    clearRect.rect = convertFixedRect(call.callDesc.rect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);
                }
                else if (call.callDesc.extendedType != DrawExtendedType::None) {
                    switch (call.callDesc.extendedType) {
                    case DrawExtendedType::VertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::VertexTestZ;
                        instanceDrawCall.vertexTestZ.vertexIndex = call.callDesc.extendedData.vertexTestZ.vertexIndex;
                        instanceDrawCall.vertexTestZ.resolutionScale = p.resolutionScale;
                        instanceDrawCall.vertexTestZ.srcIndexStart = call.meshDesc.faceIndicesStart + 3;
                        instanceDrawCall.vertexTestZ.dstIndexStart = vertexTestZFaceIndicesStart;
                        instanceDrawCall.vertexTestZ.indexCount = 0;
                        vertexTestZCallIndex = int32_t(instanceDrawCallVector.size());
                        break;
                    case DrawExtendedType::EndVertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::Unknown;
                        vertexTestZCallIndex = -1;
                        break;
                    default:
                        assert(false && "Unknown extended type.");
                        break;
                    }
                }
                else {
#               if RT_ENABLED
                    if (rtProj) {
                        instanceDrawCall.type = InstanceDrawCall::Type::Raytracing;

                        if (hitGroupVector.empty()) {
                            // TODO: Support specialized shaders.
                            //const RaytracingShaderPrograms &shaderPrograms = p.ubershadersOnly ? rtState->shaderProgramsMap.find(UberShaderHash)->second : rtState->getShaderPrograms(call.shaderDesc);
                            const RaytracingShaderPrograms &shaderPrograms = rtState->shaderProgramsMap.find(UberShaderHash)->second;
                            hitGroupVector.emplace_back(shaderPrograms.surface);
                            hitGroupVector.emplace_back(shaderPrograms.shadow);
                        }

                        auto &raytracing = instanceDrawCall.raytracing;
                        raytracing.hitGroupIndex = 0;
                        raytracing.cullDisable = !call.shaderDesc.flags.culling;

                        const interop::OtherMode &otherMode = call.callDesc.otherMode;
                        raytracing.queryMask = (otherMode.zCmp() || otherMode.zUpd()) ? DepthRayQueryMask : NoDepthRayQueryMask;
                        if (drawData.extraParams[call.callDesc.callIndex].shadowCatcherFactor > 0.0f) {
                            raytracing.queryMask |= ShadowCatcherRayQueryMask;
                        }

                        const RenderBottomLevelASMesh asMesh(indexRes->at(call.meshDesc.faceIndicesStart *IndexStride), worldPosRes->at(0), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, call.callDesc.triangleCount * 3, vertexCount, PosStride, false);
                        rtResources->addBottomLevelASMesh(asMesh);

                        // Geometry without lighting has no normals, so the path tracer would shade it with the normals
                        // of its faces. Smooth normals are computed for it instead, so low polygon models look rounder.
                        // Large draw calls are skipped, as the cost grows with the square of the triangle count.
                        static const float smoothNormalAngle = enhancementValue("RT64_RT_SMOOTH_NORMALS", 50.0f);
                        if ((smoothNormalAngle > 0.0f) && !call.callDesc.rspLit && (call.callDesc.triangleCount <= 1024)) {
                            // Models are usually drawn in several consecutive calls (like the halves of a head), so
                            // consecutive ranges of triangles are merged to weld the vertices between them too.
                            const uint32_t indexStart = call.meshDesc.faceIndicesStart;
                            const uint32_t indexCount = call.callDesc.triangleCount * 3;
                            if (!rspSmoothNormalVector.empty() && ((rspSmoothNormalVector.back().indexStart + rspSmoothNormalVector.back().indexCount) == indexStart) && ((rspSmoothNormalVector.back().indexCount + indexCount) <= (1024 * 3))) {
                                rspSmoothNormalVector.back().indexCount += indexCount;
                            }
                            else {
                                RSPSmoothNormalGenerationCB rspSmoothNormal;
                                rspSmoothNormal.indexStart = indexStart;
                                rspSmoothNormal.indexCount = indexCount;
                                rspSmoothNormal.creaseCosine = cosf(smoothNormalAngle * 3.14159265f / 180.0f);
                                rspSmoothNormalVector.push_back(rspSmoothNormal);
                            }
                        }
                    }
                    else 
#               endif
                    {
                        triangles.shaderDesc = call.shaderDesc;

                        RasterShader *gpuShader = p.ubershadersOnly ? nullptr : p.rasterShaderCache->getGPUShader(call.shaderDesc);
                        if (gpuShader != nullptr) {
                            triangles.pipeline = gpuShader->pipeline.get();
                        }
                        else {
                            const bool copyMode = (call.shaderDesc.otherMode.cycleType() == G_CYC_COPY);
                            triangles.pipeline = rasterShaderUber->getPipeline(
                                !copyMode && call.shaderDesc.otherMode.zCmp() && (call.shaderDesc.otherMode.zMode() != ZMODE_DEC),
                                !copyMode && call.shaderDesc.otherMode.zUpd(),
                                (call.shaderDesc.otherMode.cvgDst() == CVG_DST_WRAP) || (call.shaderDesc.otherMode.cvgDst() == CVG_DST_SAVE));
                        }
                        
                        triangles.faceCount = call.callDesc.triangleCount;
                        triangles.vertexTestZ = (vertexTestZCallIndex >= 0);
                        triangles.postBlendDitherNoise = false;

                        float invRatioScale = 1.0f / aspectRatioScale;
                        float horizontalMisalignment = 0.0f;
                        switch (proj.type) {
                        case Projection::Type::Perspective:
                        case Projection::Type::Orthographic: {
                            instanceDrawCall.type = InstanceDrawCall::Type::IndexedTriangles;
                            triangles.indexStart = triangles.vertexTestZ ? vertexTestZFaceIndicesStart : call.meshDesc.faceIndicesStart;
                            invRatioScale = projInvRatioScale;
                            break;
                        }
                        case Projection::Type::Rectangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RegularRect;
                            triangles.indexStart = call.meshDesc.rawVertexStart;

                            bool tileCopiesUsed = false;
                            for (uint32_t t = 0; (t < call.callDesc.tileCount) && !tileCopiesUsed; t++) {
                                tileCopiesUsed = drawData.callTiles[call.callDesc.tileIndex + t].tileCopyUsed;
                            }

                            // The call's scissor spans the whole width of the framebuffer pair scissor. The rect must not be using extended origins.
                            const bool regularOrigins = (call.callDesc.rectLeftOrigin == G_EX_ORIGIN_NONE) && (call.callDesc.rectRightOrigin == G_EX_ORIGIN_NONE);
                            const bool coversScissorWidth = regularOrigins && (call.callDesc.rect.ulx <= fbPair.scissorRect.ulx) && (call.callDesc.rect.lrx >= fbPair.scissorRect.lrx);
                            if ((tileCopiesUsed || coversScissorWidth || call.callDesc.rectAspect == G_EX_ASPECT_STRETCH) && (call.callDesc.rectAspect != G_EX_ASPECT_ADJUST)) {
                                invRatioScale = 1.0f;
                            }
                            else {
                                horizontalMisalignment = p.horizontalMisalignment;
                            }

                            RenderViewport viewportRect = convertViewportRect(call.callDesc.rect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);
                            triangles.screenScale = { viewportRect.width / framebuffer.viewport.width, viewportRect.height / framebuffer.viewport.height };
                            triangles.screenOffset.x = halfPixelOffset.x + ((viewportRect.x + viewportRect.width / 2.0f) - halfViewportSize.x) / halfViewportSize.x;
                            triangles.screenOffset.y = halfPixelOffset.y + (halfViewportSize.y - (viewportRect.y + viewportRect.height / 2.0f)) / halfViewportSize.y;

                            if (p.postBlendNoise) {
                                // Indicate if post blend dither noise should be applied.
                                bool rgbDitherNoise = (call.shaderDesc.otherMode.rgbDither() == G_CD_NOISE);
                                triangles.postBlendDitherNoise = rgbDitherNoise && !call.shaderDesc.otherMode.zCmp() && !call.shaderDesc.otherMode.zUpd();
                                triangles.postBlendDitherNoiseNegative = p.postBlendNoiseNegative;
                            }

                            break;
                        }
                        case Projection::Type::Triangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RawTriangles;
                            triangles.indexStart = call.meshDesc.rawVertexStart;
                            break;
                        }
                        case Projection::Type::None:
                        default:
                            break;
                        }

                        triangles.scissor = convertFixedRect(call.callDesc.scissorRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, int32_t(horizontalMisalignment), call.callDesc.scissorLeftOrigin, call.callDesc.scissorRightOrigin);

                        bool usesViewport = (proj.type == Projection::Type::Perspective) || (proj.type == Projection::Type::Orthographic);
                        if (usesViewport) {
                            triangles.scissor = viewportScissorIntersection(viewportClip, triangles.scissor);
                        }
                        
                        if (triangles.vertexTestZ && usesViewport) {
                            instanceDrawCallVector[vertexTestZCallIndex].vertexTestZ.indexCount += call.callDesc.triangleCount * 3;
                            vertexTestZFaceIndicesStart += call.callDesc.triangleCount * 3;
                        }

                        if ((lightingSceneIndex >= 0) && (instanceDrawCall.type == InstanceDrawCall::Type::IndexedTriangles)) {
                            const interop::OtherMode &otherMode = call.shaderDesc.otherMode;
                            const bool copyMode = (otherMode.cycleType() == G_CYC_COPY);
                            lighting->addSceneRect(uint32_t(lightingSceneIndex), triangles.scissor);
                            if (!copyMode && interop::Blender::usesAlphaBlend(otherMode)) {
                                lightingDeferredCall = true;
                            }
                            else if (!copyMode && otherMode.zUpd() && (otherMode.zMode() != ZMODE_DEC) && !triangles.vertexTestZ && !triangles.scissor.isEmpty()) {
                                const uint32_t callInstanceIndex = uint32_t(instanceDrawCallVector.size());
                                const bool alphaTested = otherMode.cvgXAlpha() || (otherMode.alphaCompare() != G_AC_NONE);
                                const bool rspLit = call.callDesc.rspLit;
                                lighting->addCaster(uint32_t(lightingSceneIndex), callInstanceIndex, alphaTested);

                                // Unlit cutouts are usually foliage drawn as flat cards.
                                uint32_t gbufferFlags = 0;
                                gbufferFlags |= alphaTested ? LIGHTING_GBUFFER_ALPHA_TESTED : 0;
                                gbufferFlags |= rspLit ? LIGHTING_GBUFFER_RSP_LIT : 0;
                                gbufferFlags |= (alphaTested && !rspLit && call.shaderDesc.flags.usesTexture0) ? LIGHTING_GBUFFER_FOLIAGE : 0;
                                gbufferFlags |= (!alphaTested && call.shaderDesc.flags.usesTexture0) ? LIGHTING_GBUFFER_BUMP : 0;
                                lighting->addGBufferDraw(uint32_t(lightingSceneIndex), callInstanceIndex, gbufferFlags);

                                // Geometry without lighting has no normals: smooth ones are computed from its faces. Large
                                // draw calls are skipped as the cost grows with the square of the triangle count, and
                                // consecutive ranges are merged to weld the models drawn in several calls.
                                if ((lightingSmoothNormalAngle > 0.0f) && !rspLit && !alphaTested && (call.callDesc.triangleCount <= lightingSmoothNormalTriangles)) {
                                    const uint32_t indexStart = call.meshDesc.faceIndicesStart;
                                    const uint32_t indexCount = call.callDesc.triangleCount * 3;
                                    if (!rspSmoothNormalVector.empty() && ((rspSmoothNormalVector.back().indexStart + rspSmoothNormalVector.back().indexCount) == indexStart) && ((rspSmoothNormalVector.back().indexCount + indexCount) <= (lightingSmoothNormalTriangles * 3))) {
                                        rspSmoothNormalVector.back().indexCount += indexCount;
                                    }
                                    else {
                                        RSPSmoothNormalGenerationCB rspSmoothNormal;
                                        rspSmoothNormal.indexStart = indexStart;
                                        rspSmoothNormal.indexCount = indexCount;
                                        rspSmoothNormal.creaseCosine = cosf(lightingSmoothNormalAngle * 3.14159265f / 180.0f);
                                        rspSmoothNormalVector.push_back(rspSmoothNormal);
                                    }
                                }
                            }
                        }
                    }
                }

                // Determine to use the draw call either in the RT scene or the raster scene.
                const uint32_t instanceIndex = static_cast<uint32_t>(instanceDrawCallVector.size());
#           if RT_ENABLED
                bool rtCall = instanceDrawCall.type == InstanceDrawCall::Type::Raytracing;
                if (rtCall) {
                    // If the current scene is not compatible, we submit it before the raster scene.
                    if (!rtProjCompatible) {
                        checkRtScene(rtScene);
                    }

                    const bool addedRasterScene = checkRasterScene(rasterScene);
                    if (rtScene.instanceIndices.empty()) {
                        float projRatioScale = 1.0f / aspectRatioScale;
                        float invRatioScale = 1.0f / aspectRatioScale;
                        const bool coversScissorWidth = (proj.scissorRect.ulx <= fbPair.scissorRect.ulx) && (proj.scissorRect.lrx >= fbPair.scissorRect.lrx);
                        if (coversScissorWidth) {
                            invRatioScale = 1.0f;
                        }
                        else {
                            projRatioScale = 1.0f;
                        }

                        rtScene.curViewMatrix = drawData.modViewTransforms[proj.transformsIndex];
                        rtScene.curProjMatrix = drawData.modProjTransforms[proj.transformsIndex];
                        rtScene.prevViewMatrix = drawData.prevViewTransforms[proj.transformsIndex];
                        rtScene.prevProjMatrix = drawData.prevProjTransforms[proj.transformsIndex];

                        const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                        // The raytraced output must cover the area the projection maps to, which is the viewport without
                        // the extended clipping ratios the game might use for the guard band.
                        const int16_t unitClipRatios[4] = { 1, 1, -1, -1 };
                        rtScene.viewport = convertViewportRect(viewport.rect(unitClipRatios), p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0.0f, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);
                        rtScene.scissor = convertFixedRect(proj.scissorRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);

                        rtScene.presetScene = p.presetScene;
                        rtScene.worldUp = p.curWorkload->worldUp;
                        rtScene.worldRight = p.curWorkload->worldRight;
                        rtScene.worldForward = p.curWorkload->worldForward;
                        rtScene.worldOrigin = p.curWorkload->worldOrigin;
                        if (float(hlslpp::dot(rtScene.worldUp, rtScene.worldUp)) < 1e-6f) {
                            rtScene.worldUp = { 0.0f, 1.0f, 0.0f };
                        }
                        rtScene.deltaTime = std::max(p.deltaTimeMs / 1000.0f, 1e-4f);
                        rtScene.screenWidth = lround(static_cast<float>(p.fbWidth) * p.resolutionScale.x);
                        rtScene.screenHeight = lround(static_cast<float>(p.fbHeight) * p.resolutionScale.y);

                        if (proj.pointLightCount > 0) {
                            rtScene.pointLights = proj.pointLights.data();
                            rtScene.lightCount = proj.pointLightCount;
                        }
                        else {
                            rtScene.pointLights = nullptr;
                            rtScene.lightCount = 0;
                        }
                    }
                    else if (rtProjCompatible && addedRasterScene) {
                        targetDrawCall.sceneIndices.pop_back();
                        const uint32_t rasterSceneIndex = static_cast<uint32_t>(targetDrawCall.rasterScenes.size() - 1);
                        const auto &rasterScene = targetDrawCall.rasterScenes[rasterSceneIndex];
                        rtScene.interleavedRasters.push_back({ rasterSceneIndex, rasterScene.instanceIndices.back(), 0, 0 });
                    }

                    rtScene.instanceIndices.push_back(instanceIndex);
                }
                else 
#           endif
                if (lightingDeferredCall) {
                    lightingDeferred.push_back(instanceIndex);
                }
                else {
                    rasterScene.instanceIndices.push_back(instanceIndex);
                }

                instanceDrawCallVector.push_back(instanceDrawCall);
                globalCallIndex++;
            }
        }

        closeLightingScene();

#   if RT_ENABLED
        checkRtScene(rtScene);
#   endif
        checkRasterScene(rasterScene);
    }

    void FramebufferRenderer::endFramebuffers(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool rtEnabled) {
        bool shaderViewRtEnabled = false;
        std::vector<BufferUploader::Upload> shaderUploads = {
            { renderIndicesVector.data(), { 0, renderIndicesVector.size() }, sizeof(interop::RenderIndices), RenderBufferFlag::STORAGE, { }, &renderIndicesBuffer},
            { &frameParams, { 0, 1 }, sizeof(interop::FrameParams), RenderBufferFlag::CONSTANT, { }, &frameParamsBuffer}
        };

#   if RT_ENABLED
        // FIXME: Add support for multiple raytracing scenes.
        Framebuffer *chosenFramebuffer = nullptr;
        RaytracingScene *chosenRtScene = nullptr;
        if (rtEnabled) {
            rtResources->updateBottomLevelASResources(worker);

            for (uint32_t i = 0; i < framebufferCount; i++) {
                RenderTargetDrawCall &targetDrawCall = framebufferVector[i].renderTargetDrawCall;
                if (!targetDrawCall.rtScenes.empty()) {
                    chosenFramebuffer = &framebufferVector[i];
                    chosenRtScene = &targetDrawCall.rtScenes[0];
                }
            }
        }

        if (chosenRtScene != nullptr) {
            const bool sizeChanged = (rtResources->screenWidth != chosenRtScene->screenWidth) || (rtResources->screenHeight != chosenRtScene->screenHeight);
            if (rtResources->updateOutputBuffers || sizeChanged) {
                rtResources->createOutputBuffers(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight);
                rtResources->updateOutputBuffers = false;
            }

            const RenderTarget *framebufferTarget = chosenFramebuffer->renderTargetDrawCall.fbStorage->colorTarget;
            interleavedRastersCount = static_cast<uint32_t>(chosenRtScene->interleavedRasters.size());

            // Debugging aid: RT64_RT_PRINT_STATS prints the composition of the raytraced scene.
            static const bool printStats = (getenv("RT64_RT_PRINT_STATS") != nullptr);
            static uint32_t printStatsCounter = 0;
            if (printStats && ((printStatsCounter++ % 120) == 0)) {
                uint32_t rtSceneCount = 0;
                for (uint32_t i = 0; i < framebufferCount; i++) {
                    rtSceneCount += uint32_t(framebufferVector[i].renderTargetDrawCall.rtScenes.size());
                }

                fprintf(stderr, "RT scene: %zu instances, %u interleaved rasters, %u RT scenes in %u framebuffers, viewport %.0f %.0f %.0f %.0f, screen %u x %u" "\n",
                    chosenRtScene->instanceIndices.size(), interleavedRastersCount, rtSceneCount, framebufferCount,
                    chosenRtScene->viewport.x, chosenRtScene->viewport.y, chosenRtScene->viewport.width, chosenRtScene->viewport.height,
                    chosenRtScene->screenWidth, chosenRtScene->screenHeight);
            }
            rtResources->updateInterleavedRenderTargets(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight, interleavedRastersCount, framebufferTarget->multisampling, framebufferTarget->usesHDR);

            for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                auto &intRaster = chosenRtScene->interleavedRasters[i];
                RenderTarget *colorTarget = rtResources->interleavedColorTargetVector[i].get();
                RenderTarget *depthTarget = rtResources->interleavedDepthTargetVector[i].get();
                intRaster.colorTextureIndex = getTextureIndex(colorTarget);
                intRaster.depthTextureIndex = getTextureIndex(depthTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(colorTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(depthTarget);
            }

            // Must have at least one element in the vector.
            if (chosenRtScene->interleavedRasters.empty()) {
                chosenRtScene->interleavedRasters.emplace_back();
            }

            shaderUploads.push_back({ &rtResources->rtParams, { 0, 1 }, sizeof(interop::RaytracingParams), RenderBufferFlag::CONSTANT, { }, &rtResources->rtParamsBuffer });
            shaderUploads.push_back({ chosenRtScene->interleavedRasters.data(), { 0, chosenRtScene->interleavedRasters.size() }, sizeof(interop::InterleavedRaster), RenderBufferFlag::STORAGE, { }, &interleavedRastersBuffer });

            updateRaytracingScene(worker, *chosenRtScene);
            shaderViewRtEnabled = true;
        }
#   endif

        lightingBuffersActive = (lighting != nullptr) && !lighting->empty();
        if (lighting != nullptr) {
            lighting->finish(worker, instanceDrawCallVector, shaderUploads);
        }

        // Table of the groups of the smooth normals: every range of indices is split into tiles of 64 triangles.
        smoothNormalGroups.clear();
        for (const RSPSmoothNormalGenerationCB &range : rspSmoothNormalVector) {
            const uint32_t triangleCount = range.indexCount / 3;
            uint32_t creaseBits;
            memcpy(&creaseBits, &range.creaseCosine, sizeof(creaseBits));
            for (uint32_t firstTriangle = 0; firstTriangle < triangleCount; firstTriangle += 64) {
                smoothNormalGroups.push_back({ range.indexStart, range.indexCount, firstTriangle, creaseBits });
            }
        }

        if (!smoothNormalGroups.empty()) {
            shaderUploads.push_back({ smoothNormalGroups.data(), { 0, smoothNormalGroups.size() }, sizeof(interop::uint4), RenderBufferFlag::STORAGE, { }, &smoothNormalGroupsBuffer });
        }

        shaderUploader->submit(worker, shaderUploads);
        updateShaderViews(worker, drawBuffers, outputBuffers, shaderViewRtEnabled);

        if (lighting != nullptr) {
            lighting->updateDescriptorSets();
        }

#   if RT_ENABLED
        // The shader binding table must be built after the descriptor sets are updated, as some backends store the
        // location of the descriptors in the table and the sets can be recreated while updating them.
        if (shaderViewRtEnabled) {
            RenderDescriptorSet *descriptorSets[] = { descCommonSet->get(), descTextureSet->get(), descTextureSet->get(), chosenFramebuffer->descRealFbSet->get() };
            rtResources->createShaderBindingTable(worker, rtState, descriptorSets, uint32_t(std::size(descriptorSets)), hitGroupVector);
            rtFramebufferIndex = uint32_t(chosenFramebuffer - framebufferVector.data());
        }
#   endif
    }

    void FramebufferRenderer::advanceFrame(bool rtEnabled) {
        frameParams.frameCount++;

#   if RT_ENABLED
        if (rtEnabled) {
            rtResources->swapBuffers = !rtResources->swapBuffers;
            rtResources->skipReprojection = false;
        }
#   endif
    }
};

/*
void RT64::View::renderIm3d() {
    if (Im3d::GetDrawListCount() > 0) {
        commandList->SetGraphicsRootSignature(worker->device->getIm3dRootSignature());

        commandList->SetDescriptorHeaps(1, &descriptorHeap);
        commandList->SetGraphicsRootDescriptorTable(0, descriptorHeap->GetGPUDescriptorHandleForHeapStart());
        commandList->RSSetViewports(1, &rtViewport);
        commandList->RSSetScissorRects(1, &rtScissor);

        unsigned int totalVertexCount = 0;
        for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
            auto &drawList = Im3d::GetDrawLists()[i];
            totalVertexCount += drawList.m_vertexCount;
        }

        if (totalVertexCount > 0) {
            // Release the previous vertex buffer if it should be bigger.
            if (!im3dVertexBuffer.IsNull() && (totalVertexCount > im3dVertexCount)) {
                im3dVertexBuffer.Release();
            }

            // Create the vertex buffer if it's empty.
            const UINT vertexBufferSize = totalVertexCount * sizeof(Im3d::VertexData);
            if (im3dVertexBuffer.IsNull()) {
                CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
                im3dVertexBuffer = worker->device->allocateResource(D3D12_HEAP_TYPE_UPLOAD, &uploadBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr);
                im3dVertexCount = totalVertexCount;
                im3dVertexBufferView.BufferLocation = im3dVertexBuffer.Get()->GetGPUVirtualAddress();
                im3dVertexBufferView.StrideInBytes = sizeof(Im3d::VertexData);
                im3dVertexBufferView.SizeInBytes = vertexBufferSize;
            }

            // Copy data to vertex buffer.
            UINT8 *pDataBegin;
            CD3DX12_RANGE readRange(0, 0);
            D3D12_CHECK(im3dVertexBuffer.Get()->Map(0, &readRange, reinterpret_cast<void **>(&pDataBegin)));
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                size_t copySize = sizeof(Im3d::VertexData) * drawList.m_vertexCount;
                memcpy(pDataBegin, drawList.m_vertexData, copySize);
                pDataBegin += copySize;
            }

            im3dVertexBuffer.Get()->Unmap(0, nullptr);

            unsigned int vertexOffset = 0;
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                commandList->IASetVertexBuffers(0, 1, &im3dVertexBufferView);
                switch (drawList.m_primType) {
                case Im3d::DrawPrimitive_Points:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStatePoint());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                    break;
                case Im3d::DrawPrimitive_Lines:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateLine());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
                    break;
                case Im3d::DrawPrimitive_Triangles:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateTriangle());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    break;
                default:
                    break;
                }

                commandList->DrawInstanced(drawList.m_vertexCount, 1, vertexOffset, 0);
                vertexOffset += drawList.m_vertexCount;
            }
        }
    }
}
*/

/*
namespace RT64 {
    class View {
    private:
        // Im3D
        AllocatedResource im3dVertexBuffer;
        D3D12_VERTEX_BUFFER_VIEW im3dVertexBufferView;
        unsigned int im3dVertexCount;
    };
};
*/
