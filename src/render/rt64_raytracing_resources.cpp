//
// RT64
//

#include "rt64_raytracing_resources.h"

#include <algorithm>
#include <cmath>

namespace RT64 {
    static const uint64_t AccelerationStructureAlignment = 256;
    static const uint32_t LumaHistogramSize = 64 * sizeof(uint32_t);

    static uint64_t alignUp(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    // RaytracingResources

    RaytracingResources::RaytracingResources(RenderWorker *worker, UserConfiguration::GraphicsAPI graphicsAPI) {
        assert(worker != nullptr);

        this->worker = worker;
        this->graphicsAPI = graphicsAPI;
    }

    RaytracingResources::~RaytracingResources() {
        releaseOutputBuffers();
    }

    void RaytracingResources::setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged) {
        rtParams.diSamples = uint32_t(std::max(rtConfig.diSamples, 0));
        rtParams.giSamples = uint32_t(std::max(rtConfig.giSamples, 0));
        rtParams.maxLights = uint32_t(std::max(rtConfig.maxLights, 0));
        rtParams.motionBlurStrength = rtConfig.motionBlurStrength;
        rtParams.motionBlurSamples = uint32_t(std::max(rtConfig.motionBlurSamples, 0));
        rtParams.visualizationMode = rtConfig.visualizationMode;

        // Debugging aid: RT64_RT_DEBUG sets the debug flags of the shaders (1 disables shadows).
        rtParams.debugFlags = (getenv("RT64_RT_DEBUG") != nullptr) ? uint32_t(strtoul(getenv("RT64_RT_DEBUG"), nullptr, 0)) : 0;

        // Debugging aid: RT64_RT_VIS selects one of the visualization modes (6 is the raw direct light).
        static const int visualizationOverride = (getenv("RT64_RT_VIS") != nullptr) ? atoi(getenv("RT64_RT_VIS")) : -1;
        if ((visualizationOverride >= 0) && (visualizationOverride < int(interop::VisualizationMode::Count))) {
            rtParams.visualizationMode = interop::VisualizationMode(visualizationOverride);
        }
        maxReflections = std::max(rtConfig.maxReflections, 0);
        denoiserEnabled = rtConfig.denoiserEnabled;
        antialiasingEnabled = rtConfig.antialiasingEnabled && (getenv("RT64_RT_NO_TAA") == nullptr);
        upscalerQualityMode = rtConfig.upscalerQualityMode;
        upscalerSharpness = rtConfig.upscalerSharpness;
        upscalerResolutionOverride = rtConfig.upscalerResolutionOverride;
        upscalerReactiveMask = rtConfig.upscalerReactiveMask;
        upscalerLockMask = rtConfig.upscalerLockMask;

        // No external upscalers are available in this build, so they always fall back to bilinear.
        const UpscaleMode newUpscalerMode = (getUpscaler(rtConfig.upscalerMode) != nullptr) ? rtConfig.upscalerMode : UpscaleMode::Bilinear;
        // Debugging aid: RT64_RT_SCALE overrides the resolution scale of the raytracing buffers.
        static const float scaleOverride = (getenv("RT64_RT_SCALE") != nullptr) ? float(atof(getenv("RT64_RT_SCALE"))) : 0.0f;
        const float newResolutionScale = std::clamp((scaleOverride > 0.0f) ? scaleOverride : rtConfig.resolutionScale, 0.1f, 2.0f);
        if (resolutionChanged || (newUpscalerMode != upscalerMode) || (newResolutionScale != resolutionScale)) {
            updateOutputBuffers = true;
        }

        upscalerMode = newUpscalerMode;
        resolutionScale = newResolutionScale;
    }

    void RaytracingResources::createOutputBuffers(RenderWorker *worker, uint32_t screenWidth, uint32_t screenHeight) {
        assert(worker != nullptr);

        releaseOutputBuffers();

        this->screenWidth = std::max(screenWidth, 1U);
        this->screenHeight = std::max(screenHeight, 1U);
        textureWidth = std::max(uint32_t(std::lround(this->screenWidth * resolutionScale)), 8U);
        textureHeight = std::max(uint32_t(std::lround(this->screenHeight * resolutionScale)), 8U);
        upscaleActive = false;

        rtParams.resolution.x = float(textureWidth);
        rtParams.resolution.y = float(textureHeight);
        rtParams.resolution.z = float(this->screenWidth);
        rtParams.resolution.w = float(this->screenHeight);

        RenderDevice *device = worker->device;
        auto createStorageTexture = [&](RenderFormat format, uint32_t width, uint32_t height, const char *name) {
            std::unique_ptr<RenderTexture> texture = device->createTexture(RenderTextureDesc::Texture2D(width, height, 1, format, RenderTextureFlag::STORAGE | RenderTextureFlag::UNORDERED_ACCESS));
            texture->setName(name);
            return texture;
        };

        const uint32_t w = textureWidth;
        const uint32_t h = textureHeight;
        viewDirectionTexture = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT View Direction");
        shadingPositionTexture = createStorageTexture(RenderFormat::R32G32B32A32_FLOAT, w, h, "RT Shading Position");
        shadingNormalTexture = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Shading Normal");
        shadingSpecularTexture = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Shading Specular");
        diffuseTexture = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Diffuse");
        instanceIdTexture = createStorageTexture(RenderFormat::R32_SINT, w, h, "RT Instance ID");
        reflectionTexture = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Reflection");
        refractionTexture = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Refraction");
        transparentTexture = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Transparent");
        flowTexture = createStorageTexture(RenderFormat::R16G16_FLOAT, w, h, "RT Flow");
        reactiveMaskTexture = createStorageTexture(RenderFormat::R8_UNORM, w, h, "RT Reactive Mask");
        lockMaskTexture = createStorageTexture(RenderFormat::R8_UNORM, w, h, "RT Lock Mask");
        for (uint32_t i = 0; i < 2; i++) {
            directLightTexture[i] = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Direct Light");
            indirectLightTexture[i] = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Indirect Light");
            filteredDirectLightTexture[i] = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Filtered Direct Light");
            filteredIndirectLightTexture[i] = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Filtered Indirect Light");
            normalRoughnessTexture[i] = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Normal Roughness");
            antialiasedTexture[i] = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, w, h, "RT Antialiased Output");
            depthTexture[i] = createStorageTexture(RenderFormat::R32_FLOAT, w, h, "RT Depth");

            outputTexture[i] = device->createTexture(RenderTextureDesc::ColorTarget(w, h, RenderFormat::R32G32B32A32_FLOAT));
            outputTexture[i]->setName("RT Output");

            const RenderTexture *colorAttachment = outputTexture[i].get();
            outputFramebuffer[i] = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1));
        }

        const uint32_t downscaledWidth = std::max(w / 8, 1U);
        const uint32_t downscaledHeight = std::max(h / 8, 1U);
        downscaledOutputTexture = createStorageTexture(RenderFormat::R32G32B32A32_FLOAT, downscaledWidth, downscaledHeight, "RT Downscaled Output");
        bloomWidth = std::max(w / 4, 1U);
        bloomHeight = std::max(h / 4, 1U);
        for (uint32_t i = 0; i < 2; i++) {
            bloomTexture[i] = createStorageTexture(RenderFormat::R16G16B16A16_FLOAT, bloomWidth, bloomHeight, "RT Bloom");
        }

        lumaAverageTexture = createStorageTexture(RenderFormat::R32_FLOAT, 1, 1, "RT Luma Average");
        lumaHistogramBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(LumaHistogramSize, RenderBufferFlag::UNORDERED_ACCESS | RenderBufferFlag::STORAGE));
        lumaHistogramBuffer->setName("RT Luma Histogram");

        // Hit buffers store every surface found along the primary and secondary rays for every pixel.
        const uint64_t hitCount = uint64_t(w) * uint64_t(h) * MaxHitQueries;
        const RenderBufferFlags hitBufferFlags = RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS | RenderBufferFlag::FORMATTED;
        hitVelocityDistanceBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(hitCount * 16, hitBufferFlags));
        hitColorBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(hitCount * 4, hitBufferFlags));
        hitNormalFogBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(hitCount * 8, hitBufferFlags));
        hitInstanceIdBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(hitCount * 2, hitBufferFlags));
        hitVelocityDistanceBufferView = hitVelocityDistanceBuffer->createBufferFormattedView(RenderFormat::R32G32B32A32_FLOAT);
        hitColorBufferView = hitColorBuffer->createBufferFormattedView(RenderFormat::R8G8B8A8_UNORM);
        hitNormalFogBufferView = hitNormalFogBuffer->createBufferFormattedView(RenderFormat::R16G16B16A16_FLOAT);
        hitInstanceIdBufferView = hitInstanceIdBuffer->createBufferFormattedView(RenderFormat::R16_UINT);

        transitionOutputBuffers = true;
        skipReprojection = true;
        shaderSetsDirty = true;
    }

    void RaytracingResources::releaseOutputBuffers() {
        postProcessSet.reset();
        antialiasingSets[0].reset();
        antialiasingSets[1].reset();
        for (uint32_t i = 0; i < 3; i++) {
            bloomSets[i].reset();
        }

        lumaSetSet.reset();
        lumaClearSet.reset();
        lumaAvgSet.reset();
        lumaSet.reset();
        downscaleSet.reset();
        composeSet.reset();
        indirectFilterSets[0].reset();
        indirectFilterSets[1].reset();

        viewDirectionTexture.reset();
        shadingPositionTexture.reset();
        shadingNormalTexture.reset();
        shadingSpecularTexture.reset();
        diffuseTexture.reset();
        instanceIdTexture.reset();
        reflectionTexture.reset();
        refractionTexture.reset();
        transparentTexture.reset();
        flowTexture.reset();
        reactiveMaskTexture.reset();
        lockMaskTexture.reset();
        for (uint32_t i = 0; i < 2; i++) {
            directLightTexture[i].reset();
            indirectLightTexture[i].reset();
            filteredDirectLightTexture[i].reset();
            filteredIndirectLightTexture[i].reset();
            normalRoughnessTexture[i].reset();
            antialiasedTexture[i].reset();
            depthTexture[i].reset();
            outputFramebuffer[i].reset();
            outputTexture[i].reset();
        }

        upscaledOutputTexture.reset();
        downscaledOutputTexture.reset();
        bloomTexture[0].reset();
        bloomTexture[1].reset();
        lumaAverageTexture.reset();
        lumaHistogramBuffer.reset();
        hitVelocityDistanceBufferView.reset();
        hitColorBufferView.reset();
        hitNormalFogBufferView.reset();
        hitInstanceIdBufferView.reset();
        hitVelocityDistanceBuffer.reset();
        hitColorBuffer.reset();
        hitNormalFogBuffer.reset();
        hitInstanceIdBuffer.reset();
    }

    void RaytracingResources::updateShaderSets(RenderWorker *worker, const ShaderLibrary *shaderLibrary) {
        assert(shaderLibrary != nullptr);

        if (diffuseTexture == nullptr) {
            return;
        }

        RenderDevice *device = worker->device;
        const SamplerLibrary &samplerLibrary = shaderLibrary->samplerLibrary;
        if (shaderSetsDirty) {
            for (uint32_t i = 0; i < 2; i++) {
                indirectFilterSets[i] = std::make_unique<GaussianFilterDescriptorSet>(samplerLibrary, device);
            }

            composeSet = std::make_unique<RaytracingComposeDescriptorSet>(samplerLibrary, device);
            downscaleSet = std::make_unique<BicubicScalingDescriptorSet>(samplerLibrary, device);
            lumaSet = std::make_unique<LuminanceHistogramDescriptorSet>(device);
            lumaAvgSet = std::make_unique<HistogramAverageDescriptorSet>(device);
            lumaClearSet = std::make_unique<HistogramClearDescriptorSet>(device);
            lumaSetSet = std::make_unique<HistogramSetDescriptorSet>(device);
            postProcessSet = std::make_unique<PostProcessDescriptorSet>(samplerLibrary, device);

            // Each set accumulates the output of one frame on top of the history written by the other one.
            for (uint32_t i = 0; i < 2; i++) {
                antialiasingSets[i] = std::make_unique<TemporalAADescriptorSet>(samplerLibrary, device);
                TemporalAADescriptorSet *aaSet = antialiasingSets[i].get();
                aaSet->setTexture(aaSet->gCurrent, outputTexture[i].get(), RenderTextureLayout::SHADER_READ);
                aaSet->setTexture(aaSet->gHistory, antialiasedTexture[i ^ 1].get(), RenderTextureLayout::SHADER_READ);
                aaSet->setTexture(aaSet->gFlow, flowTexture.get(), RenderTextureLayout::SHADER_READ);
                aaSet->setTexture(aaSet->gOutput, antialiasedTexture[i].get(), RenderTextureLayout::GENERAL);
            }

            // The gaussian filter ping-pongs between both filtered indirect textures.
            for (uint32_t i = 0; i < 2; i++) {
                GaussianFilterDescriptorSet *filterSet = indirectFilterSets[i].get();
                filterSet->setTexture(filterSet->gInput, filteredIndirectLightTexture[i].get(), RenderTextureLayout::SHADER_READ);
                filterSet->setTexture(filterSet->gOutput, filteredIndirectLightTexture[i ^ 1].get(), RenderTextureLayout::GENERAL);
            }

            composeSet->setTexture(composeSet->gFlow, flowTexture.get(), RenderTextureLayout::GENERAL);
            composeSet->setTexture(composeSet->gDiffuse, diffuseTexture.get(), RenderTextureLayout::SHADER_READ);
            composeSet->setTexture(composeSet->gDirectLight, filteredDirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ);
            composeSet->setTexture(composeSet->gIndirectLight, filteredIndirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ);
            composeSet->setTexture(composeSet->gReflection, reflectionTexture.get(), RenderTextureLayout::SHADER_READ);
            composeSet->setTexture(composeSet->gRefraction, refractionTexture.get(), RenderTextureLayout::SHADER_READ);
            composeSet->setTexture(composeSet->gTransparent, transparentTexture.get(), RenderTextureLayout::SHADER_READ);

            downscaleSet->setTexture(downscaleSet->gOutput, downscaledOutputTexture.get(), RenderTextureLayout::GENERAL);
            lumaSet->setTexture(lumaSet->HDRTexture, downscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ);
            lumaSet->setBuffer(lumaSet->LuminanceHistogram, lumaHistogramBuffer.get(), LumaHistogramSize);
            lumaAvgSet->setBuffer(lumaAvgSet->LuminanceHistogram, lumaHistogramBuffer.get(), LumaHistogramSize);
            lumaAvgSet->setTexture(lumaAvgSet->LuminanceOutput, lumaAverageTexture.get(), RenderTextureLayout::GENERAL);
            lumaClearSet->setBuffer(lumaClearSet->LuminanceHistogram, lumaHistogramBuffer.get(), LumaHistogramSize);
            lumaSetSet->setTexture(lumaSetSet->LuminanceOutput, lumaAverageTexture.get(), RenderTextureLayout::GENERAL);

            postProcessSet->setTexture(postProcessSet->gFlow, flowTexture.get(), RenderTextureLayout::SHADER_READ);
            postProcessSet->setTexture(postProcessSet->gLumaAvg, lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ);
            postProcessSet->setTexture(postProcessSet->gBloom, bloomTexture[0].get(), RenderTextureLayout::SHADER_READ);

            for (uint32_t i = 0; i < 3; i++) {
                bloomSets[i] = std::make_unique<BloomDescriptorSet>(samplerLibrary, device);
            }

            bloomSets[0]->setTexture(bloomSets[0]->gOutput, bloomTexture[0].get(), RenderTextureLayout::GENERAL);
            bloomSets[1]->setTexture(bloomSets[1]->gInput, bloomTexture[0].get(), RenderTextureLayout::SHADER_READ);
            bloomSets[1]->setTexture(bloomSets[1]->gOutput, bloomTexture[1].get(), RenderTextureLayout::GENERAL);
            bloomSets[2]->setTexture(bloomSets[2]->gInput, bloomTexture[1].get(), RenderTextureLayout::SHADER_READ);
            bloomSets[2]->setTexture(bloomSets[2]->gOutput, bloomTexture[0].get(), RenderTextureLayout::GENERAL);
            shaderSetsDirty = false;
        }

        // These change every frame depending on which of the output textures is active.
        RenderTexture *rtOutputCur = outputTexture[swapBuffers ? 1 : 0].get();
        downscaleSet->setTexture(downscaleSet->gInput, rtOutputCur, RenderTextureLayout::SHADER_READ);
        postProcessSet->setBuffer(postProcessSet->RtParams, rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        RenderTexture *postProcessInput = rtOutputCur;
        if (upscaleActive && (upscaledOutputTexture != nullptr)) {
            postProcessInput = upscaledOutputTexture.get();
        }
        else if (antialiasingEnabled) {
            postProcessInput = antialiasedTexture[swapBuffers ? 1 : 0].get();
        }

        postProcessSet->setTexture(postProcessSet->gInput, postProcessInput, RenderTextureLayout::SHADER_READ);
        bloomSets[0]->setTexture(bloomSets[0]->gInput, postProcessInput, RenderTextureLayout::SHADER_READ);
    }

    void RaytracingResources::updateMultisampling() {
        // The raytracing output is always single sampled. Interleaved targets are recreated when the multisampling changes.
        interleavedWidth = 0;
        interleavedHeight = 0;
    }

    void RaytracingResources::updateInterleavedRenderTargets(RenderWorker *worker, uint32_t width, uint32_t height, uint32_t count, const RenderMultisampling &multisampling, bool usesHDR) {
        const bool sameSettings = (interleavedWidth == width) && (interleavedHeight == height) && (interleavedUsesHDR == usesHDR) && (interleavedMultisampling.sampleCount == multisampling.sampleCount);
        if (!sameSettings) {
            interleavedFramebufferStorageVector.clear();
            interleavedColorTargetVector.clear();
            interleavedDepthTargetVector.clear();
            interleavedWidth = width;
            interleavedHeight = height;
            interleavedUsesHDR = usesHDR;
            interleavedMultisampling = multisampling;
        }

        while (interleavedColorTargetVector.size() < count) {
            std::unique_ptr<RenderTarget> colorTarget = std::make_unique<RenderTarget>(0, Framebuffer::Type::Color, multisampling, usesHDR);
            std::unique_ptr<RenderTarget> depthTarget = std::make_unique<RenderTarget>(0, Framebuffer::Type::Depth, multisampling, usesHDR);
            colorTarget->setupColor(worker, width, height);
            depthTarget->setupDepth(worker, width, height);

            std::unique_ptr<RenderFramebufferStorage> fbStorage = std::make_unique<RenderFramebufferStorage>();
            fbStorage->setup(worker->device, RenderFramebufferKey(), colorTarget.get(), depthTarget.get());
            interleavedColorTargetVector.emplace_back(std::move(colorTarget));
            interleavedDepthTargetVector.emplace_back(std::move(depthTarget));
            interleavedFramebufferStorageVector.emplace_back(std::move(fbStorage));
        }
    }

    void RaytracingResources::resetBottomLevelAS() {
        bottomLevelASVector.clear();
        bottomLevelASCount = 0;
        topLevelASPending = false;
    }

    void RaytracingResources::addBottomLevelASMesh(const RenderBottomLevelASMesh &mesh) {
        BottomLevelAS blas;
        blas.mesh = mesh;
        bottomLevelASVector.emplace_back(std::move(blas));
        bottomLevelASCount++;
    }

    void RaytracingResources::updateBottomLevelASResources(RenderWorker *worker) {
        if (bottomLevelASVector.empty()) {
            return;
        }

        RenderDevice *device = worker->device;
        uint64_t totalBufferSize = 0;
        uint64_t totalScratchSize = 0;
        for (BottomLevelAS &blas : bottomLevelASVector) {
            device->setBottomLevelASBuildInfo(blas.buildInfo, &blas.mesh, 1, true, false);
            blas.bufferOffset = totalBufferSize;
            blas.scratchOffset = totalScratchSize;
            totalBufferSize += alignUp(blas.buildInfo.accelerationStructureSize, AccelerationStructureAlignment);
            totalScratchSize += alignUp(blas.buildInfo.scratchSize, AccelerationStructureAlignment);
        }

        if ((bottomLevelASBuffer == nullptr) || (bottomLevelASBufferSize < totalBufferSize)) {
            bottomLevelASBufferSize = alignUp(totalBufferSize * 3 / 2, AccelerationStructureAlignment);
            bottomLevelASBuffer = device->createBuffer(RenderBufferDesc::AccelerationStructureBuffer(bottomLevelASBufferSize));
            bottomLevelASBuffer->setName("RT Bottom Level AS");
        }

        if ((scratchBuffer == nullptr) || (scratchBufferSize < totalScratchSize)) {
            scratchBufferSize = alignUp(totalScratchSize * 3 / 2, AccelerationStructureAlignment);
            scratchBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(scratchBufferSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH | RenderBufferFlag::UNORDERED_ACCESS));
            scratchBuffer->setName("RT Scratch");
        }

        for (BottomLevelAS &blas : bottomLevelASVector) {
            const RenderAccelerationStructureDesc asDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, bottomLevelASBuffer->at(blas.bufferOffset), blas.buildInfo.accelerationStructureSize);
            blas.accelerationStructure = device->createAccelerationStructure(asDesc);
        }
    }

    void RaytracingResources::submitBottomLevelASCreation(RenderWorker *worker) {
        if (bottomLevelASVector.empty()) {
            return;
        }

        RenderCommandList *commandList = worker->commandList.get();
        for (const BottomLevelAS &blas : bottomLevelASVector) {
            commandList->buildBottomLevelAS(blas.accelerationStructure.get(), scratchBuffer->at(blas.scratchOffset), blas.buildInfo);
        }

        // Wait for all bottom level structures to be built before building the top level one.
        commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(bottomLevelASBuffer.get(), RenderBufferAccess::READ | RenderBufferAccess::WRITE));
        commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(scratchBuffer.get(), RenderBufferAccess::READ | RenderBufferAccess::WRITE));
    }

    void RaytracingResources::updateTopLevelASResources(RenderWorker *worker, const std::vector<InstanceDrawCall> &instanceDrawCalls, const std::vector<uint32_t> &instanceIndices) {
        RenderDevice *device = worker->device;

        // Bottom level structures were added in the same order as the raytracing draw calls, so the index of each one
        // is the amount of raytracing draw calls that came before it.
        thread_local std::vector<int32_t> blasIndices;
        blasIndices.assign(instanceDrawCalls.size(), -1);
        int32_t blasIndex = 0;
        for (size_t i = 0; i < instanceDrawCalls.size(); i++) {
            if (instanceDrawCalls[i].type == InstanceDrawCall::Type::Raytracing) {
                blasIndices[i] = blasIndex++;
            }
        }

        topLevelASInstances.clear();
        for (uint32_t instanceIndex : instanceIndices) {
            assert(instanceIndex < instanceDrawCalls.size());
            const int32_t index = blasIndices[instanceIndex];
            if ((index < 0) || (uint32_t(index) >= bottomLevelASVector.size())) {
                continue;
            }

            const BottomLevelAS &blas = bottomLevelASVector[index];
            if (blas.buildInfo.primitiveCount == 0) {
                continue;
            }

            const auto &raytracing = instanceDrawCalls[instanceIndex].raytracing;
            const uint32_t hitGroupContribution = raytracing.hitGroupIndex * 2;
            topLevelASInstances.emplace_back(bottomLevelASBuffer->at(blas.bufferOffset), instanceIndex, raytracing.queryMask, hitGroupContribution, raytracing.cullDisable, RenderAffineTransform());
        }

        // Use a dummy instance with no geometry if the scene is empty so the structure is still valid.
        device->setTopLevelASBuildInfo(topLevelASBuildInfo, topLevelASInstances.data(), uint32_t(topLevelASInstances.size()), true, false);

        if ((topLevelASBuffer == nullptr) || (topLevelASBufferSize < topLevelASBuildInfo.accelerationStructureSize)) {
            topLevelASBufferSize = alignUp(std::max<uint64_t>(topLevelASBuildInfo.accelerationStructureSize * 3 / 2, 4096), AccelerationStructureAlignment);
            topLevelASBuffer = device->createBuffer(RenderBufferDesc::AccelerationStructureBuffer(topLevelASBufferSize));
            topLevelASBuffer->setName("RT Top Level AS");
        }

        const uint64_t instancesSize = std::max<uint64_t>(topLevelASBuildInfo.instancesBufferData.size(), 256);
        if ((topLevelASInstancesBuffer == nullptr) || (topLevelASInstancesBufferSize < instancesSize)) {
            topLevelASInstancesBufferSize = alignUp(instancesSize * 3 / 2, AccelerationStructureAlignment);
            topLevelASInstancesBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(topLevelASInstancesBufferSize, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
            topLevelASInstancesBuffer->setName("RT Top Level AS Instances");
        }

        if (!topLevelASBuildInfo.instancesBufferData.empty()) {
            void *instancesData = topLevelASInstancesBuffer->map();
            memcpy(instancesData, topLevelASBuildInfo.instancesBufferData.data(), topLevelASBuildInfo.instancesBufferData.size());
            topLevelASInstancesBuffer->unmap();
        }

        // The scratch buffer of the top level structure is placed after the bottom level scratch data.
        const uint64_t requiredScratch = alignUp(topLevelASBuildInfo.scratchSize, AccelerationStructureAlignment);
        if ((scratchBuffer == nullptr) || (scratchBufferSize < requiredScratch)) {
            scratchBufferSize = alignUp(std::max(requiredScratch, scratchBufferSize) * 3 / 2, AccelerationStructureAlignment);
            scratchBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(scratchBufferSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH | RenderBufferFlag::UNORDERED_ACCESS));
            scratchBuffer->setName("RT Scratch");
        }

        const RenderAccelerationStructureDesc asDesc(RenderAccelerationStructureType::TOP_LEVEL, topLevelASBuffer->at(0), topLevelASBuildInfo.accelerationStructureSize);
        topLevelAS = device->createAccelerationStructure(asDesc);
        topLevelASPending = true;
    }

    void RaytracingResources::submitTopLevelASCreation(RenderWorker *worker) {
        if (!topLevelASPending) {
            return;
        }

        RenderCommandList *commandList = worker->commandList.get();
        commandList->buildTopLevelAS(topLevelAS.get(), scratchBuffer->at(0), topLevelASInstancesBuffer->at(0), topLevelASBuildInfo);
        commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(topLevelASBuffer.get(), RenderBufferAccess::READ | RenderBufferAccess::WRITE));
        topLevelASPending = false;
    }

    void RaytracingResources::createShaderBindingTable(RenderWorker *worker, const RaytracingState *rtState, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount, const std::vector<RenderPipelineProgram> &hitGroups) {
        assert(rtState != nullptr);

        const RenderShaderBindingGroups groups(
            RenderShaderBindingGroup(rtState->rayGenPrograms.data(), uint32_t(rtState->rayGenPrograms.size())),
            RenderShaderBindingGroup(rtState->missPrograms.data(), uint32_t(rtState->missPrograms.size())),
            RenderShaderBindingGroup(hitGroups.data(), uint32_t(hitGroups.size()))
        );

        RenderDevice *device = worker->device;
        device->setShaderBindingTableInfo(shaderBindingTableInfo, groups, rtState->pipeline.get(), descriptorSets, descriptorSetCount);

        const uint64_t tableSize = shaderBindingTableInfo.tableBufferData.size();
        if ((shaderBindingTableBuffer == nullptr) || (shaderBindingTableBufferSize < tableSize)) {
            shaderBindingTableBufferSize = alignUp(std::max<uint64_t>(tableSize, 1024), 256);
            shaderBindingTableBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(shaderBindingTableBufferSize, RenderBufferFlag::SHADER_BINDING_TABLE));
            shaderBindingTableBuffer->setName("RT Shader Binding Table");
        }

        void *tableData = shaderBindingTableBuffer->map();
        memcpy(tableData, shaderBindingTableInfo.tableBufferData.data(), tableSize);
        shaderBindingTableBuffer->unmap();
    }

    void RaytracingResources::updateLightsBuffer(RenderWorker *worker, const RaytracingScene &rtScene) {
        const uint32_t lightCount = (rtScene.pointLights != nullptr) ? rtScene.lightCount : 0;
        const uint32_t requiredCapacity = std::max(lightCount, 1U);
        if ((lightsBuffer == nullptr) || (lightsBufferCapacity < requiredCapacity)) {
            lightsBufferCapacity = std::max(requiredCapacity, 16U);
            lightsBuffer = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::PointLight) * lightsBufferCapacity, RenderBufferFlag::STORAGE));
            lightsBuffer->setName("RT Lights");
        }

        if (lightCount > 0) {
            void *lightsData = lightsBuffer->map();
            memcpy(lightsData, rtScene.pointLights, sizeof(interop::PointLight) * lightCount);
            lightsBuffer->unmap();
        }

        rtParams.lightsCount = lightCount;

        static const bool printLights = (getenv("RT64_RT_PRINT_LIGHTS") != nullptr);
        if (printLights) {
            static uint32_t printCount = 0;
            if ((printCount++ % 120) == 0) {
                fprintf(stderr, "RT lights: %u\n", lightCount);
                for (uint32_t i = 0; i < lightCount; i++) {
                    const interop::PointLight &l = rtScene.pointLights[i];
                    fprintf(stderr, "  pos %.1f %.1f %.1f col %.2f %.2f %.2f radius %.1f\n", l.position.x, l.position.y, l.position.z, l.diffuseColor.x, l.diffuseColor.y, l.diffuseColor.z, l.attenuationRadius);
                }
            }
        }
    }

    Upscaler *RaytracingResources::getUpscaler(UpscaleMode mode) const {
        // FSR, DLSS and XeSS are not part of this build.
        return nullptr;
    }
};
