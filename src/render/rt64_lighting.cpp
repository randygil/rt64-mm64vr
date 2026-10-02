//
// RT64
//

#include "rt64_lighting.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include "shaders/FullScreenVS.hlsl.spirv.h"
#include "shaders/LightingCopyPS.hlsl.spirv.h"
#include "shaders/LightingCopyPSMS.hlsl.spirv.h"
#include "shaders/LightingComposePS.hlsl.spirv.h"
#include "shaders/LightingComposePSMS.hlsl.spirv.h"
#include "shaders/LightingGBufferVS.hlsl.spirv.h"
#include "shaders/LightingGBufferPS.hlsl.spirv.h"
#include "shaders/LightingGBufferPSMS.hlsl.spirv.h"
#include "shaders/LightingGBufferFoliagePS.hlsl.spirv.h"
#include "shaders/LightingGBufferFoliagePSMS.hlsl.spirv.h"
#include "shaders/LightingShadowPS.hlsl.spirv.h"
#include "shaders/LightingShadowVS.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/FullScreenVS.hlsl.dxil.h"
#   include "shaders/LightingCopyPS.hlsl.dxil.h"
#   include "shaders/LightingCopyPSMS.hlsl.dxil.h"
#   include "shaders/LightingComposePS.hlsl.dxil.h"
#   include "shaders/LightingComposePSMS.hlsl.dxil.h"
#   include "shaders/LightingGBufferVS.hlsl.dxil.h"
#   include "shaders/LightingGBufferPS.hlsl.dxil.h"
#   include "shaders/LightingGBufferPSMS.hlsl.dxil.h"
#   include "shaders/LightingGBufferFoliagePS.hlsl.dxil.h"
#   include "shaders/LightingGBufferFoliagePSMS.hlsl.dxil.h"
#   include "shaders/LightingShadowPS.hlsl.dxil.h"
#   include "shaders/LightingShadowVS.hlsl.dxil.h"
#elif defined(__APPLE__)
#   include "shaders/FullScreenVS.hlsl.metal.h"
#   include "shaders/LightingCopyPS.hlsl.metal.h"
#   include "shaders/LightingCopyPSMS.hlsl.metal.h"
#   include "shaders/LightingComposePS.hlsl.metal.h"
#   include "shaders/LightingComposePSMS.hlsl.metal.h"
#   include "shaders/LightingGBufferVS.hlsl.metal.h"
#   include "shaders/LightingGBufferPS.hlsl.metal.h"
#   include "shaders/LightingGBufferPSMS.hlsl.metal.h"
#   include "shaders/LightingGBufferFoliagePS.hlsl.metal.h"
#   include "shaders/LightingGBufferFoliagePSMS.hlsl.metal.h"
#   include "shaders/LightingShadowPS.hlsl.metal.h"
#   include "shaders/LightingShadowVS.hlsl.metal.h"
#endif

#include "rt64_shader_library.h"
#include "rt64_tuning.h"

#ifdef _WIN32
#   define LIGHTING_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? NAME##BlobDXIL : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? sizeof(NAME##BlobDXIL) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#elif defined(__APPLE__)
#   define LIGHTING_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? NAME##BlobMSL : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? sizeof(NAME##BlobMSL) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#else
#   define LIGHTING_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#endif

namespace RT64 {
    static bool lightingEnabledFromEnvironment() {
        const char *value = getenv("RT64_LIGHTING");
        return (value != nullptr) && (atoi(value) != 0);
    }

    static std::atomic<bool> rasterLightingEnabled = lightingEnabledFromEnvironment();
    static std::atomic<int> rasterLightingQuality = 1;

    void setRasterLightingEnabled(bool enabled) {
        rasterLightingEnabled = enabled;
    }

    bool isRasterLightingEnabled() {
        return rasterLightingEnabled;
    }

    void setRasterLightingQuality(int quality) {
        rasterLightingQuality = std::clamp(quality, 0, 3);
    }

    int getRasterLightingQuality() {
        return rasterLightingQuality;
    }

    static float dot3(const hlslpp::float3 &a, const hlslpp::float3 &b) {
        return float(hlslpp::dot(a, b));
    }

    static float length3(const hlslpp::float3 &a) {
        return float(hlslpp::length(a));
    }

    static hlslpp::float3 normalize3(const hlslpp::float3 &a, const hlslpp::float3 &fallback) {
        const float l = length3(a);
        return (l > 1e-8f) ? (a / l) : fallback;
    }

    static hlslpp::float3 transformPoint(const hlslpp::float4x4 &m, const hlslpp::float3 &p) {
        const hlslpp::float4 r = hlslpp::mul(hlslpp::float4(p, 1.0f), m);
        return r.xyz / float(r.w);
    }

    float computeStableShadowMatrix(const hlslpp::float3 &sunDirection, const hlslpp::float3 &sphereCenter, float sphereRadius, float casterDistance, uint32_t mapSize,
        const hlslpp::float3 &worldRight, const hlslpp::float3 &worldUp, const hlslpp::float3 &worldForward, const hlslpp::float3 &worldOrigin, interop::float4x4 &shadowMatrix, float &depthRange)
    {
        // Everything is computed in a world aligned basis so the texels of the shadow map stay fixed in the world.
        auto toWorldDirection = [&](const hlslpp::float3 &d) {
            return hlslpp::float3(dot3(d, worldRight), dot3(d, worldUp), dot3(d, worldForward));
        };

        const hlslpp::float3 centerWorld = toWorldDirection(sphereCenter - worldOrigin);
        const hlslpp::float3 lightForward = normalize3(-toWorldDirection(sunDirection), hlslpp::float3(0.0f, -1.0f, 0.0f));
        const hlslpp::float3 referenceUp = (fabsf(float(lightForward.y)) > 0.99f) ? hlslpp::float3(0.0f, 0.0f, 1.0f) : hlslpp::float3(0.0f, 1.0f, 0.0f);
        const hlslpp::float3 lightRight = normalize3(hlslpp::cross(referenceUp, lightForward), hlslpp::float3(1.0f, 0.0f, 0.0f));
        const hlslpp::float3 lightUp = hlslpp::cross(lightForward, lightRight);
        const float texelSize = (2.0f * sphereRadius) / float(mapSize);
        const float centerX = floorf(dot3(centerWorld, lightRight) / texelSize) * texelSize;
        const float centerY = floorf(dot3(centerWorld, lightUp) / texelSize) * texelSize;
        const float centerZ = dot3(centerWorld, lightForward);
        const float zNear = -(sphereRadius + casterDistance);
        const float zFar = sphereRadius;
        depthRange = zFar - zNear;

        // World space to the clip space of the light, as a row vector matrix.
        hlslpp::float4x4 lightMatrix(
            float(lightRight.x) / sphereRadius, float(lightUp.x) / sphereRadius, float(lightForward.x) / depthRange, 0.0f,
            float(lightRight.y) / sphereRadius, float(lightUp.y) / sphereRadius, float(lightForward.y) / depthRange, 0.0f,
            float(lightRight.z) / sphereRadius, float(lightUp.z) / sphereRadius, float(lightForward.z) / depthRange, 0.0f,
            -centerX / sphereRadius, -centerY / sphereRadius, (-centerZ - zNear) / depthRange, 1.0f);

        // Space of the geometry to the world aligned basis.
        hlslpp::float4x4 worldMatrix(
            float(worldRight.x), float(worldUp.x), float(worldForward.x), 0.0f,
            float(worldRight.y), float(worldUp.y), float(worldForward.y), 0.0f,
            float(worldRight.z), float(worldUp.z), float(worldForward.z), 0.0f,
            -dot3(worldOrigin, worldRight), -dot3(worldOrigin, worldUp), -dot3(worldOrigin, worldForward), 1.0f);

        shadowMatrix = hlslpp::mul(worldMatrix, lightMatrix);
        return texelSize;
    }

    // LightingRenderer

    LightingRenderer::LightingRenderer(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat) {
        assert(device != nullptr);
        assert(shaderLibrary != nullptr);

        this->device = device;
        this->shaderFormat = shaderFormat;

        RenderSamplerDesc samplerDesc;
        samplerDesc.minFilter = RenderFilter::LINEAR;
        samplerDesc.magFilter = RenderFilter::LINEAR;
        samplerDesc.mipmapMode = RenderMipmapMode::NEAREST;
        samplerDesc.addressU = RenderTextureAddressMode::CLAMP;
        samplerDesc.addressV = RenderTextureAddressMode::CLAMP;
        samplerDesc.addressW = RenderTextureAddressMode::CLAMP;
        samplerDesc.comparisonEnabled = true;
        samplerDesc.comparisonFunc = RenderComparisonFunction::LESS_EQUAL;
        shadowSampler = device->createSampler(samplerDesc);

        // The shadow casters are drawn with the descriptors of the raster shaders.
        {
            FramebufferRendererDescriptorCommonSet descriptorCommonSet(shaderLibrary->samplerLibrary, device->getCapabilities().raytracing);
            FramebufferRendererDescriptorTextureSet descriptorTextureSet;
            FramebufferRendererDescriptorFramebufferSet descriptorFramebufferSet;
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin(false, true);
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingShadowCB), RenderShaderStageFlag::VERTEX | RenderShaderStageFlag::PIXEL);
            layoutBuilder.addDescriptorSet(descriptorCommonSet);
            layoutBuilder.addDescriptorSet(descriptorTextureSet);
            layoutBuilder.addDescriptorSet(descriptorTextureSet);
            layoutBuilder.addDescriptorSet(descriptorFramebufferSet);
            layoutBuilder.end();
            shadowPipelineLayout = layoutBuilder.create(device);

            static const RenderInputSlot InputSlots[3] = {
                RenderInputSlot(0, RenderFormatSize(RenderFormat::R32G32B32A32_FLOAT)),
                RenderInputSlot(1, RenderFormatSize(RenderFormat::R32G32_FLOAT)),
                RenderInputSlot(2, RenderFormatSize(RenderFormat::R32G32B32A32_FLOAT))
            };

            static const RenderInputElement InputElements[3] = {
                RenderInputElement("POSITION", 0, 0, RenderFormat::R32G32B32A32_FLOAT, 0, 0),
                RenderInputElement("TEXCOORD", 0, 1, RenderFormat::R32G32_FLOAT, 1, 0),
                RenderInputElement("COLOR", 0, 2, RenderFormat::R32G32B32A32_FLOAT, 2, 0)
            };

            std::unique_ptr<RenderShader> vertexShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingShadowVS, "VSMain", shaderFormat));
            std::unique_ptr<RenderShader> pixelShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingShadowPS, "PSMain", shaderFormat));
            RenderGraphicsPipelineDesc pipelineDesc;
            pipelineDesc.pipelineLayout = shadowPipelineLayout.get();
            pipelineDesc.vertexShader = vertexShader.get();
            pipelineDesc.inputSlots = InputSlots;
            pipelineDesc.inputSlotsCount = uint32_t(std::size(InputSlots));
            pipelineDesc.inputElements = InputElements;
            pipelineDesc.inputElementsCount = uint32_t(std::size(InputElements));
            pipelineDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            pipelineDesc.cullMode = RenderCullMode::NONE;
            pipelineDesc.depthEnabled = true;
            pipelineDesc.depthWriteEnabled = true;
            pipelineDesc.depthFunction = RenderComparisonFunction::LESS;

            // Casters between the sun and the near plane of the light are flattened onto it instead of being clipped.
            pipelineDesc.depthClipEnabled = false;
            pipelineDesc.slopeScaledDepthBias = 1.5f;
            pipelineDesc.depthTargetFormat = RenderFormat::D32_FLOAT;
            pipelineDesc.renderTargetCount = 0;
            shadowOpaquePipeline = device->createGraphicsPipeline(pipelineDesc);

            pipelineDesc.pixelShader = pixelShader.get();
            shadowAlphaPipeline = device->createGraphicsPipeline(pipelineDesc);

            // The normal buffer is drawn with the same inputs as the raster shaders.
            layoutBuilder.begin(false, true);
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingGBufferCB), RenderShaderStageFlag::VERTEX | RenderShaderStageFlag::PIXEL);
            layoutBuilder.addDescriptorSet(descriptorCommonSet);
            layoutBuilder.addDescriptorSet(descriptorTextureSet);
            layoutBuilder.addDescriptorSet(descriptorTextureSet);
            layoutBuilder.addDescriptorSet(descriptorFramebufferSet);
            layoutBuilder.end();
            gbufferPipelineLayout = layoutBuilder.create(device);

            // Foliage normals need the index of the primitive in the pixel shader, which requires geometry shader support.
            foliageNormalsSupported = device->getCapabilities().geometryShader;

            std::unique_ptr<RenderShader> gbufferVertexShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingGBufferVS, "VSMain", shaderFormat));
            std::unique_ptr<RenderShader> gbufferPixelShaders[2][2];
            gbufferPixelShaders[0][0] = device->createShader(LIGHTING_SHADER_INPUTS(LightingGBufferPS, "PSMain", shaderFormat));
            gbufferPixelShaders[0][1] = device->createShader(LIGHTING_SHADER_INPUTS(LightingGBufferPSMS, "PSMain", shaderFormat));
            if (foliageNormalsSupported) {
                gbufferPixelShaders[1][0] = device->createShader(LIGHTING_SHADER_INPUTS(LightingGBufferFoliagePS, "PSMain", shaderFormat));
                gbufferPixelShaders[1][1] = device->createShader(LIGHTING_SHADER_INPUTS(LightingGBufferFoliagePSMS, "PSMain", shaderFormat));
            }

            RenderGraphicsPipelineDesc gbufferDesc;
            gbufferDesc.pipelineLayout = gbufferPipelineLayout.get();
            gbufferDesc.vertexShader = gbufferVertexShader.get();
            gbufferDesc.inputSlots = InputSlots;
            gbufferDesc.inputSlotsCount = uint32_t(std::size(InputSlots));
            gbufferDesc.inputElements = InputElements;
            gbufferDesc.inputElementsCount = uint32_t(std::size(InputElements));
            gbufferDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            gbufferDesc.cullMode = RenderCullMode::NONE;
            gbufferDesc.depthClipEnabled = false;
            gbufferDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_UNORM;
            gbufferDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            gbufferDesc.renderTargetCount = 1;
            for (uint32_t foliage = 0; foliage < 2; foliage++) {
                for (uint32_t multisampling = 0; multisampling < 2; multisampling++) {
                    if (gbufferPixelShaders[foliage][multisampling] != nullptr) {
                        gbufferDesc.pixelShader = gbufferPixelShaders[foliage][multisampling].get();
                        gbufferPipelines[foliage][multisampling] = device->createGraphicsPipeline(gbufferDesc);
                    }
                }
            }
        }

        // Composition of the lighting over the color target.
        {
            LightingComposeDescriptorSet descriptorSet(shadowSampler.get());
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingComposeCB), RenderShaderStageFlag::PIXEL);
            layoutBuilder.addDescriptorSet(descriptorSet);
            layoutBuilder.end();
            composePipelineLayout = layoutBuilder.create(device);
            fullScreenVertexShader = device->createShader(LIGHTING_SHADER_INPUTS(FullScreenVS, "VSMain", shaderFormat));
            composePixelShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingComposePS, "PSMain", shaderFormat));
            composePixelShaderMS = device->createShader(LIGHTING_SHADER_INPUTS(LightingComposePSMS, "PSMain", shaderFormat));
        }

        // Copy of the color target for the passes that read it.
        {
            LightingCopyDescriptorSet descriptorSet;
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addDescriptorSet(descriptorSet);
            layoutBuilder.end();
            copyPipelineLayout = layoutBuilder.create(device);
            copyPixelShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingCopyPS, "PSMain", shaderFormat));
            copyPixelShaderMS = device->createShader(LIGHTING_SHADER_INPUTS(LightingCopyPSMS, "PSMain", shaderFormat));
        }

        sky = std::make_unique<LightingSky>(device, shaderLibrary, shaderFormat);
        postEffects = std::make_unique<PostEffects>(device, shaderLibrary, shaderFormat);
    }

    static float lightingTime() {
        static const auto startTime = std::chrono::steady_clock::now();
        return std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
    }

    const RenderTexture *LightingRenderer::copyColor(RenderWorker *worker, uint32_t sceneIndex) {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        RenderTarget *colorTarget = scene.colorTarget;
        const bool multisampling = (colorTarget->multisampling.sampleCount > 1);
        if ((colorCopyTexture == nullptr) || (colorCopyWidth < colorTarget->width) || (colorCopyHeight < colorTarget->height) || (colorCopyFormat != colorTarget->format)) {
            colorCopyFramebuffer.reset();
            colorCopyTexture.reset();
            colorCopyWidth = std::max(colorCopyWidth, colorTarget->width);
            colorCopyHeight = std::max(colorCopyHeight, colorTarget->height);
            colorCopyFormat = colorTarget->format;
            colorCopyTexture = device->createTexture(RenderTextureDesc::ColorTarget(colorCopyWidth, colorCopyHeight, colorCopyFormat));
            colorCopyTexture->setName("Lighting Color Copy");
            const RenderTexture *colorAttachment = colorCopyTexture.get();
            colorCopyFramebuffer = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1));
        }

        std::unique_ptr<RenderPipeline> &pipeline = copyPipelines[{ multisampling ? 1U : 0U, colorTarget->format }];
        if (pipeline == nullptr) {
            RenderGraphicsPipelineDesc pipelineDesc;
            pipelineDesc.pipelineLayout = copyPipelineLayout.get();
            pipelineDesc.vertexShader = fullScreenVertexShader.get();
            pipelineDesc.pixelShader = multisampling ? copyPixelShaderMS.get() : copyPixelShader.get();
            pipelineDesc.renderTargetFormat[0] = colorTarget->format;
            pipelineDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            pipelineDesc.renderTargetCount = 1;
            pipelineDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            pipelineDesc.cullMode = RenderCullMode::NONE;
            pipeline = device->createGraphicsPipeline(pipelineDesc);
        }

        while (copySets.size() <= sceneIndex) {
            copySets.emplace_back(std::make_unique<LightingCopyDescriptorSet>(device));
        }

        LightingCopyDescriptorSet *copySet = copySets[sceneIndex].get();
        copySet->setTexture(copySet->gInput, colorTarget->texture.get(), RenderTextureLayout::SHADER_READ, colorTarget->textureView.get());
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(colorCopyTexture.get(), RenderTextureLayout::COLOR_WRITE)
        });

        worker->commandList->setFramebuffer(colorCopyFramebuffer.get());
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(colorCopyWidth), float(colorCopyHeight)));
        worker->commandList->setScissors(scene.rect);
        worker->commandList->setPipeline(pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(copyPipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(copySet->get(), 0);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
            RenderTextureBarrier(colorCopyTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE)
        });

        return colorCopyTexture.get();
    }

    void LightingRenderer::recordSky(RenderWorker *worker, uint32_t sceneIndex) {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        if (!scene.hasSun || scene.rect.isEmpty() || !sky->enabled()) {
            return;
        }

        LightingSkyDesc skyDesc;
        skyDesc.colorTarget = scene.colorTarget;
        skyDesc.depthTarget = scene.depthTarget;
        skyDesc.rect = scene.rect;
        skyDesc.lighting = &scene.params;
        skyDesc.sceneColor = copyColor(worker, sceneIndex);
        skyDesc.time = lightingTime();
        sky->record(worker, skyDesc);
    }

    void LightingRenderer::recordPostEffects(RenderWorker *worker, uint32_t sceneIndex) {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        if (scene.rect.isEmpty() || !postEffects->enabled()) {
            return;
        }

        PostEffectsSceneDesc postDesc;
        postDesc.colorTarget = scene.colorTarget;
        postDesc.depthTarget = scene.depthTarget;
        postDesc.rect = scene.rect;
        postDesc.lighting = &scene.params;
        postDesc.sceneColor = copyColor(worker, sceneIndex);
        postDesc.time = lightingTime();
        postEffects->record(worker, postDesc);
    }

    LightingRenderer::~LightingRenderer() { }

    void LightingRenderer::reset() {
        gbufferEnabled = (enhancementValue("RT64_LIGHT_GBUFFER", 1.0f) > 0.0f);
        scenes.clear();
        casters.clear();
        paramsVector.clear();
        shadowMapActive = false;
        shadowMapRendered = false;
    }

    uint32_t LightingRenderer::addScene(const LightingSceneDesc &desc, RenderTarget *colorTarget, RenderTarget *depthTarget) {
        Scene scene;
        scene.colorTarget = colorTarget;
        scene.depthTarget = depthTarget;
        scene.rect = RenderRect(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
        scene.viewport = desc.viewport;
        scene.screenScale = desc.screenScale;
        scene.screenOffset = desc.screenOffset;

        interop::LightingParams &params = scene.params;
        memset(&params, 0, sizeof(params));
        params.viewProj = desc.viewProj;
        const hlslpp::float4x4 invViewProj = hlslpp::inverse(desc.viewProj);
        params.invViewProj = invViewProj;
        params.shadowMatrix = interop::float4x4::identity();
        params.pixelToClip = desc.pixelToClip;
        params.depthToClip = desc.depthToClip;

        // The camera sits where all the rays of the projection meet.
        const hlslpp::float4x4 invView = hlslpp::inverse(desc.view);
        scene.cameraPosition = hlslpp::float3(float(invView[3][0]), float(invView[3][1]), float(invView[3][2]));
        params.cameraPosition = hlslpp::float4(scene.cameraPosition, 1.0f);

        // Directions of the center and the corners of the frustum, used to fit the shadow map around what's visible.
        const float MidDepth = 0.5f;
        scene.viewDirection = normalize3(transformPoint(invViewProj, hlslpp::float3(0.0f, 0.0f, MidDepth)) - scene.cameraPosition, hlslpp::float3(0.0f, 0.0f, 1.0f));
        scene.frustumSlope = 0.0f;
        const float Corners[4][2] = { { -1.0f, -1.0f }, { 1.0f, -1.0f }, { -1.0f, 1.0f }, { 1.0f, 1.0f } };
        for (const auto &corner : Corners) {
            const hlslpp::float3 cornerDirection = normalize3(transformPoint(invViewProj, hlslpp::float3(corner[0], corner[1], MidDepth)) - scene.cameraPosition, scene.viewDirection);
            const float axial = std::max(dot3(cornerDirection, scene.viewDirection), 1e-3f);
            const float lateral = length3(cornerDirection - scene.viewDirection * axial);
            scene.frustumSlope = std::max(scene.frustumSlope, lateral / axial);
        }

        // Orthonormal basis of the world.
        scene.worldUp = normalize3(desc.worldUp, hlslpp::float3(0.0f, 1.0f, 0.0f));
        hlslpp::float3 right = desc.worldRight - scene.worldUp * dot3(desc.worldRight, scene.worldUp);
        if (dot3(right, right) < 1e-6f) {
            right = (fabsf(float(scene.worldUp.x)) < 0.9f) ? hlslpp::float3(1.0f, 0.0f, 0.0f) : hlslpp::float3(0.0f, 0.0f, 1.0f);
            right = right - scene.worldUp * dot3(right, scene.worldUp);
        }

        scene.worldRight = normalize3(right, hlslpp::float3(1.0f, 0.0f, 0.0f));
        scene.worldForward = hlslpp::cross(scene.worldRight, scene.worldUp);
        if (dot3(scene.worldForward, desc.worldForward) < 0.0f) {
            scene.worldForward = -scene.worldForward;
        }

        scene.worldOrigin = (float(desc.worldOrigin.w) > 0.0f) ? desc.worldOrigin.xyz : hlslpp::float3(0.0f, 0.0f, 0.0f);
        params.worldRight = hlslpp::float4(scene.worldRight, 0.0f);
        params.worldUp = hlslpp::float4(scene.worldUp, 0.0f);
        params.worldForward = hlslpp::float4(scene.worldForward, 0.0f);
        params.worldOrigin = hlslpp::float4(scene.worldOrigin, (float(desc.worldOrigin.w) > 0.0f) ? 1.0f : 0.0f);

        // The sun is the light placed much further away than anything else. Other lights are carried with the camera.
        const float sunStrength = enhancementValue("RT64_LIGHT_SUN", 0.9f);
        for (uint32_t i = 0; i < desc.lightCount; i++) {
            const interop::PointLight &light = desc.lights[i];
            const hlslpp::float3 position(light.position.x, light.position.y, light.position.z);
            const float distance = length3(position);
            if (distance > 1e6f) {
                if (!scene.hasSun) {
                    scene.hasSun = true;
                    scene.sunDirection = position / distance;
                    const float maxComponent = std::max(std::max(light.diffuseColor.x, light.diffuseColor.y), std::max(light.diffuseColor.z, 1e-6f));
                    params.sunDirection = hlslpp::float4(scene.sunDirection, 1.0f);
                    params.sunColor = hlslpp::float4(light.diffuseColor.x / maxComponent, light.diffuseColor.y / maxComponent, light.diffuseColor.z / maxComponent, 0.0f) * sunStrength;
                }
            }
            else if (float(params.pointLightPosition.w) <= 0.0f) {
                const float pointStrength = enhancementValue("RT64_LIGHT_POINT", 0.55f);
                params.pointLightPosition = hlslpp::float4(position, light.attenuationRadius);
                params.pointLightColor = hlslpp::float4(light.diffuseColor.x, light.diffuseColor.y, light.diffuseColor.z, 0.0f) * pointStrength;
            }
        }

        if (scene.hasSun) {
            params.ambientColor = hlslpp::float4(enhancementValue("RT64_LIGHT_SKY_R", 0.62f), enhancementValue("RT64_LIGHT_SKY_G", 0.66f), enhancementValue("RT64_LIGHT_SKY_B", 0.74f), 0.0f);
            params.groundColor = hlslpp::float4(enhancementValue("RT64_LIGHT_GROUND_R", 0.50f), enhancementValue("RT64_LIGHT_GROUND_G", 0.47f), enhancementValue("RT64_LIGHT_GROUND_B", 0.42f), 0.0f);
        }
        else {
            const float indoorAmbient = enhancementValue("RT64_LIGHT_INDOOR_AMBIENT", 0.8f);
            params.ambientColor = hlslpp::float4(indoorAmbient, indoorAmbient, indoorAmbient, 0.0f);
            const float indoorGround = indoorAmbient * enhancementValue("RT64_LIGHT_INDOOR_GROUND", 0.85f);
            params.groundColor = hlslpp::float4(indoorGround, indoorGround, indoorGround, 0.0f);
        }

        params.fog = hlslpp::float4(desc.fogMul, desc.fogOffset, desc.fogEnabled ? 1.0f : 0.0f, 0.0f);
        params.lightingParams = hlslpp::float4(enhancementValue("RT64_LIGHT_STRENGTH", 1.0f), enhancementValue("RT64_LIGHT_EXPOSURE", 1.0f), enhancementValue("RT64_LIGHT_WRAP", 0.5f), enhancementValue("RT64_LIGHT_SHADING", 1.0f));
        params.foliageParams = hlslpp::float4(enhancementValue("RT64_LIGHT_FOLIAGE_WRAP", 0.8f), enhancementValue("RT64_LIGHT_FOLIAGE_TRANSLUCENCY", 0.6f), enhancementValue("RT64_LIGHT_FOLIAGE_SHADOW", 0.35f), enhancementValue("RT64_LIGHT_FOLIAGE_SHADOW_OFFSET", 1.0f));
        params.settings.x = uint32_t(enhancementValue("RT64_LIGHT_DEBUG", 0.0f));
        params.settings.y = uint32_t(getRasterLightingQuality());
        params.settings.z = (depthTarget != nullptr) ? depthTarget->multisampling.sampleCount : 1;

        const uint32_t sceneIndex = uint32_t(scenes.size());
        scenes.emplace_back(scene);
        return sceneIndex;
    }

    void LightingRenderer::addSceneRect(uint32_t sceneIndex, const RenderRect &rect) {
        assert(sceneIndex < scenes.size());
        if (rect.isEmpty()) {
            return;
        }

        RenderRect &sceneRect = scenes[sceneIndex].rect;
        sceneRect.left = std::min(sceneRect.left, rect.left);
        sceneRect.top = std::min(sceneRect.top, rect.top);
        sceneRect.right = std::max(sceneRect.right, rect.right);
        sceneRect.bottom = std::max(sceneRect.bottom, rect.bottom);
    }

    void LightingRenderer::addCaster(uint32_t instanceIndex, bool alphaTested) {
        casters.push_back({ instanceIndex, alphaTested });
    }

    void LightingRenderer::addGBufferDraw(uint32_t sceneIndex, uint32_t instanceIndex, uint32_t flags) {
        assert(sceneIndex < scenes.size());
        if (!gbufferEnabled) {
            return;
        }

        if (!foliageNormalsSupported) {
            flags &= ~LIGHTING_GBUFFER_FOLIAGE;
        }

        scenes[sceneIndex].gbufferDraws.push_back({ instanceIndex, flags });
    }

    void LightingRenderer::createNormalBuffer(RenderWorker *worker, uint32_t width, uint32_t height) {
        normalFramebuffer.reset();
        normalBuffer.reset();
        normalBuffer = device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_UNORM));
        normalBuffer->setName("Lighting Normal Buffer");
        const RenderTexture *colorAttachment = normalBuffer.get();
        normalFramebuffer = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1));
        normalBufferWidth = width;
        normalBufferHeight = height;
    }

    void LightingRenderer::createShadowMap(RenderWorker *worker, uint32_t size) {
        shadowFramebuffer.reset();
        shadowMapView.reset();
        shadowMap.reset();
        shadowMap = device->createTexture(RenderTextureDesc::DepthTarget(size, size, RenderFormat::D32_FLOAT));
        shadowMap->setName("Lighting Shadow Map");
        shadowMapView = shadowMap->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::D32_FLOAT));
        shadowFramebuffer = device->createFramebuffer(RenderFramebufferDesc(nullptr, 0, shadowMap.get()));
        shadowMapSize = size;
        shadowMapNeedsTransition = true;
    }

    void LightingRenderer::finish(RenderWorker *worker, std::vector<BufferUploader::Upload> &uploads) {
        if (scenes.empty()) {
            return;
        }

        static const uint32_t ShadowMapSizes[] = { 1024, 2048, 2048, 4096 };
        const uint32_t desiredSize = uint32_t(enhancementValue("RT64_LIGHT_SHADOW_SIZE", float(ShadowMapSizes[getRasterLightingQuality()])));
        const uint32_t mapSize = std::clamp(desiredSize, 256U, 8192U);
        if ((shadowMap == nullptr) || (shadowMapSize != mapSize)) {
            createShadowMap(worker, mapSize);
        }

        // The normal buffer covers the biggest color target of the frame.
        uint32_t normalWidth = 1;
        uint32_t normalHeight = 1;
        for (const Scene &scene : scenes) {
            normalWidth = std::max(normalWidth, scene.colorTarget->width);
            normalHeight = std::max(normalHeight, scene.colorTarget->height);
        }

        if ((normalBuffer == nullptr) || (normalBufferWidth < normalWidth) || (normalBufferHeight < normalHeight)) {
            createNormalBuffer(worker, std::max(normalWidth, normalBufferWidth), std::max(normalHeight, normalBufferHeight));
        }

        // The shadow map follows the first scene with a sun, usually the main view.
        const Scene *sunScene = nullptr;
        for (const Scene &scene : scenes) {
            if (scene.hasSun) {
                sunScene = &scene;
                break;
            }
        }

        const float shadowStrength = enhancementValue("RT64_LIGHT_SHADOW_STRENGTH", 1.0f);
        float texelSize = 1.0f;
        float depthRange = 1.0f;
        shadowMapActive = (sunScene != nullptr) && !casters.empty() && (shadowStrength > 0.0f);
        if (shadowMapActive) {
            // Bounding sphere of the part of the view frustum that receives shadows, with its center along the view
            // direction where the sphere is the smallest. Its size only depends on the field of view, so it stays the
            // same while the camera turns and the texels keep their size.
            const float shadowDistance = enhancementValue("RT64_LIGHT_SHADOW_DISTANCE", 4000.0f);
            const float casterDistance = enhancementValue("RT64_LIGHT_SHADOW_CASTER_DISTANCE", 6000.0f);
            const float k = std::max(sunScene->frustumSlope, 0.1f);
            const float centerDistance = std::min(shadowDistance * (1.0f + k * k) * 0.5f, shadowDistance);
            float radius = std::max(centerDistance, sqrtf((shadowDistance - centerDistance) * (shadowDistance - centerDistance) + (shadowDistance * k) * (shadowDistance * k)));
            radius = ceilf(radius / 16.0f) * 16.0f;

            const hlslpp::float3 center = sunScene->cameraPosition + sunScene->viewDirection * centerDistance;
            texelSize = computeStableShadowMatrix(sunScene->sunDirection, center, radius, casterDistance, mapSize,
                sunScene->worldRight, sunScene->worldUp, sunScene->worldForward, sunScene->worldOrigin, shadowMatrix, depthRange);
        }

        paramsVector.clear();
        for (Scene &scene : scenes) {
            interop::LightingParams &params = scene.params;
            params.viewportRect = hlslpp::float4(float(scene.rect.left), float(scene.rect.top), float(scene.rect.right), float(scene.rect.bottom));
            if (shadowMapActive && scene.hasSun) {
                params.shadowMatrix = shadowMatrix;
                params.shadowParams = hlslpp::float4(texelSize, enhancementValue("RT64_LIGHT_SHADOW_BIAS", 1.0f) * texelSize / depthRange,
                    enhancementValue("RT64_LIGHT_SHADOW_NORMAL_OFFSET", 1.5f), enhancementValue("RT64_LIGHT_SHADOW_SOFTNESS", 1.0f));
                params.shadowMapParams = hlslpp::float4(1.0f / mapSize, 1.0f / mapSize, shadowStrength, 0.0f);
            }
            else {
                params.shadowMapParams = hlslpp::float4(1.0f / mapSize, 1.0f / mapSize, 0.0f, 0.0f);
            }

            paramsVector.emplace_back(params);
        }

        uploads.push_back({ paramsVector.data(), { 0, paramsVector.size() }, sizeof(interop::LightingParams), RenderBufferFlag::STORAGE, { }, &paramsBuffer });

        static const bool printScenes = (getenv("RT64_LIGHT_PRINT") != nullptr);
        static uint32_t printCounter = 0;
        if (printScenes && ((printCounter++ % 120) == 0)) {
            for (uint32_t i = 0; i < uint32_t(scenes.size()); i++) {
                const Scene &scene = scenes[i];
                fprintf(stderr, "Lighting scene %u: rect %d %d %d %d, sun %d, camera %.1f %.1f %.1f, view %.2f %.2f %.2f, slope %.2f, depthToClip %.4f %.4f %.6f, casters %zu, texel %.2f\n",
                    i, scene.rect.left, scene.rect.top, scene.rect.right, scene.rect.bottom, scene.hasSun ? 1 : 0,
                    float(scene.cameraPosition.x), float(scene.cameraPosition.y), float(scene.cameraPosition.z),
                    float(scene.viewDirection.x), float(scene.viewDirection.y), float(scene.viewDirection.z), scene.frustumSlope,
                    float(scene.params.depthToClip.x), float(scene.params.depthToClip.y), float(scene.params.depthToClip.z), casters.size(), texelSize);
            }
        }
    }

    void LightingRenderer::updateDescriptorSets() {
        while (composeSets.size() < scenes.size()) {
            composeSets.emplace_back(std::make_unique<LightingComposeDescriptorSet>(shadowSampler.get(), device));
        }

        for (uint32_t i = 0; i < uint32_t(scenes.size()); i++) {
            const Scene &scene = scenes[i];
            LightingComposeDescriptorSet *set = composeSets[i].get();
            set->setBuffer(set->gLightingParams, paramsBuffer.get(), RenderBufferStructuredView(sizeof(interop::LightingParams)));
            set->setTexture(set->gDepth, scene.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, scene.depthTarget->textureView.get());
            set->setTexture(set->gShadowMap, shadowMap.get(), RenderTextureLayout::DEPTH_READ, shadowMapView.get());
            set->setTexture(set->gNormalBuffer, normalBuffer.get(), RenderTextureLayout::SHADER_READ);
        }
    }

    void LightingRenderer::recordShadowMap(RenderWorker *worker, RenderDescriptorSet *commonSet, RenderDescriptorSet *textureSet, RenderDescriptorSet *framebufferSet,
        const RenderVertexBufferView *vertexViews, const RenderInputSlot *inputSlots, uint32_t vertexViewCount, const RenderIndexBufferView *indexView,
        const std::vector<InstanceDrawCall> &instanceDrawCalls)
    {
        if (shadowMap == nullptr) {
            return;
        }

        // The shadow map must always be readable, even on frames that don't draw it.
        if (shadowMapNeedsTransition) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(shadowMap.get(), RenderTextureLayout::DEPTH_WRITE));
            worker->commandList->setFramebuffer(shadowFramebuffer.get());
            worker->commandList->clearDepth(true, 1.0f);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(shadowMap.get(), RenderTextureLayout::DEPTH_READ));
            shadowMapNeedsTransition = false;
        }

        if (!shadowMapActive || shadowMapRendered) {
            return;
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(shadowMap.get(), RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(shadowFramebuffer.get());
        worker->commandList->clearDepth(true, 1.0f);
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(shadowMapSize), float(shadowMapSize)));
        worker->commandList->setScissors(RenderRect(0, 0, int32_t(shadowMapSize), int32_t(shadowMapSize)));
        worker->commandList->setGraphicsPipelineLayout(shadowPipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(commonSet, 0);
        worker->commandList->setGraphicsDescriptorSet(textureSet, 1);
        worker->commandList->setGraphicsDescriptorSet(textureSet, 2);
        worker->commandList->setGraphicsDescriptorSet(framebufferSet, 3);
        worker->commandList->setVertexBuffers(0, vertexViews, vertexViewCount, inputSlots);
        worker->commandList->setIndexBuffer(indexView);

        interop::LightingShadowCB shadowCB;
        shadowCB.shadowMatrix = shadowMatrix;
        shadowCB.padding = { 0, 0, 0 };
        // Opaque casters don't need their draw call parameters, so consecutive ranges of indices are drawn together.
        const RenderPipeline *previousPipeline = nullptr;
        uint32_t pendingIndexStart = 0;
        uint32_t pendingIndexCount = 0;
        auto flushPending = [&]() {
            if (pendingIndexCount > 0) {
                worker->commandList->drawIndexedInstanced(pendingIndexCount, 1, pendingIndexStart, 0, 0);
                pendingIndexCount = 0;
            }
        };

        for (const Caster &caster : casters) {
            const InstanceDrawCall &drawCall = instanceDrawCalls[caster.instanceIndex];
            if ((drawCall.type != InstanceDrawCall::Type::IndexedTriangles) || (drawCall.triangles.faceCount == 0)) {
                continue;
            }

            const RenderPipeline *pipeline = caster.alphaTested ? shadowAlphaPipeline.get() : shadowOpaquePipeline.get();
            const uint32_t indexCount = drawCall.triangles.faceCount * 3;
            if (!caster.alphaTested && (pipeline == previousPipeline) && (pendingIndexCount > 0) && ((pendingIndexStart + pendingIndexCount) == drawCall.triangles.indexStart)) {
                pendingIndexCount += indexCount;
                continue;
            }

            flushPending();
            if (pipeline != previousPipeline) {
                worker->commandList->setPipeline(pipeline);
                previousPipeline = pipeline;
            }

            shadowCB.renderIndex = caster.instanceIndex;
            worker->commandList->setGraphicsPushConstants(0, &shadowCB);
            if (caster.alphaTested) {
                worker->commandList->drawIndexedInstanced(indexCount, 1, drawCall.triangles.indexStart, 0, 0);
            }
            else {
                pendingIndexStart = drawCall.triangles.indexStart;
                pendingIndexCount = indexCount;
            }
        }

        flushPending();

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(shadowMap.get(), RenderTextureLayout::DEPTH_READ));
        shadowMapRendered = true;
    }

    void LightingRenderer::recordGBuffer(RenderWorker *worker, uint32_t sceneIndex, RenderDescriptorSet *commonSet, RenderDescriptorSet *textureSet, RenderDescriptorSet *framebufferSet,
        const RenderVertexBufferView *vertexViews, const RenderInputSlot *inputSlots, uint32_t vertexViewCount, const RenderIndexBufferView *indexView,
        const std::vector<InstanceDrawCall> &instanceDrawCalls)
    {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        if (scene.rect.isEmpty() || (normalBuffer == nullptr)) {
            return;
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(normalBuffer.get(), RenderTextureLayout::COLOR_WRITE));
        worker->commandList->setFramebuffer(normalFramebuffer.get());
        worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f), &scene.rect, 1);
        if (!scene.gbufferDraws.empty()) {
            const bool multisampling = (scene.depthTarget->multisampling.sampleCount > 1);
            worker->commandList->setViewports(scene.viewport);
            worker->commandList->setScissors(scene.rect);
            worker->commandList->setGraphicsPipelineLayout(gbufferPipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(commonSet, 0);
            worker->commandList->setGraphicsDescriptorSet(textureSet, 1);
            worker->commandList->setGraphicsDescriptorSet(textureSet, 2);
            worker->commandList->setGraphicsDescriptorSet(framebufferSet, 3);
            worker->commandList->setVertexBuffers(0, vertexViews, vertexViewCount, inputSlots);
            worker->commandList->setIndexBuffer(indexView);

            interop::LightingGBufferCB gbufferCB;
            gbufferCB.cameraPosition = scene.params.cameraPosition;
            gbufferCB.screenScale = scene.screenScale;
            gbufferCB.screenOffset = scene.screenOffset;
            gbufferCB.padding = 0;
            // Draw calls without flags only need the vertices, so consecutive ranges of indices are drawn together.
            const RenderPipeline *previousPipeline = nullptr;
            uint32_t pendingIndexStart = 0;
            uint32_t pendingIndexCount = 0;
            auto flushPending = [&]() {
                if (pendingIndexCount > 0) {
                    worker->commandList->drawIndexedInstanced(pendingIndexCount, 1, pendingIndexStart, 0, 0);
                    pendingIndexCount = 0;
                }
            };

            for (const GBufferDraw &draw : scene.gbufferDraws) {
                const InstanceDrawCall &drawCall = instanceDrawCalls[draw.instanceIndex];
                if ((drawCall.type != InstanceDrawCall::Type::IndexedTriangles) || (drawCall.triangles.faceCount == 0)) {
                    continue;
                }

                const bool foliage = (draw.flags & LIGHTING_GBUFFER_FOLIAGE) != 0;
                const RenderPipeline *pipeline = gbufferPipelines[foliage ? 1 : 0][multisampling ? 1 : 0].get();
                const uint32_t indexCount = drawCall.triangles.faceCount * 3;
                if ((draw.flags == 0) && (pipeline == previousPipeline) && (pendingIndexCount > 0) && ((pendingIndexStart + pendingIndexCount) == drawCall.triangles.indexStart)) {
                    pendingIndexCount += indexCount;
                    continue;
                }

                flushPending();
                if (pipeline != previousPipeline) {
                    worker->commandList->setPipeline(pipeline);
                    previousPipeline = pipeline;
                }

                gbufferCB.renderIndex = draw.instanceIndex;
                gbufferCB.indexStart = drawCall.triangles.indexStart;
                gbufferCB.flags = draw.flags;
                worker->commandList->setGraphicsPushConstants(0, &gbufferCB);
                if (draw.flags == 0) {
                    pendingIndexStart = drawCall.triangles.indexStart;
                    pendingIndexCount = indexCount;
                }
                else {
                    worker->commandList->drawIndexedInstanced(indexCount, 1, drawCall.triangles.indexStart, 0, 0);
                }
            }

            flushPending();
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(normalBuffer.get(), RenderTextureLayout::SHADER_READ));
    }

    LightingRenderer::ComposePipelines &LightingRenderer::getComposePipelines(const RenderMultisampling &multisampling, RenderFormat format) {
        const std::pair<uint32_t, RenderFormat> key = { multisampling.sampleCount, format };
        auto it = composePipelines.find(key);
        if (it != composePipelines.end()) {
            return it->second;
        }

        ComposePipelines &pipelines = composePipelines[key];
        RenderGraphicsPipelineDesc pipelineDesc;
        pipelineDesc.pipelineLayout = composePipelineLayout.get();
        pipelineDesc.vertexShader = fullScreenVertexShader.get();
        pipelineDesc.pixelShader = (multisampling.sampleCount > 1) ? composePixelShaderMS.get() : composePixelShader.get();
        pipelineDesc.renderTargetFormat[0] = format;
        pipelineDesc.renderTargetCount = 1;
        pipelineDesc.multisampling = multisampling;
        pipelineDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
        pipelineDesc.cullMode = RenderCullMode::NONE;

        // The alpha channel holds the coverage of the RDP and must be kept.
        const uint8_t ColorMask = uint8_t(RenderColorWriteEnable::RED) | uint8_t(RenderColorWriteEnable::GREEN) | uint8_t(RenderColorWriteEnable::BLUE);
        RenderBlendDesc &blendDesc = pipelineDesc.renderTargetBlend[0];
        blendDesc.blendEnabled = true;
        blendDesc.srcBlend = RenderBlend::DEST_COLOR;
        blendDesc.dstBlend = RenderBlend::SRC_COLOR;
        blendDesc.blendOp = RenderBlendOperation::ADD;
        blendDesc.srcBlendAlpha = RenderBlend::ZERO;
        blendDesc.dstBlendAlpha = RenderBlend::ONE;
        blendDesc.blendOpAlpha = RenderBlendOperation::ADD;
        blendDesc.renderTargetWriteMask = ColorMask;
        pipelines.multiply = device->createGraphicsPipeline(pipelineDesc);

        blendDesc = RenderBlendDesc::Copy();
        blendDesc.renderTargetWriteMask = ColorMask;
        pipelines.copy = device->createGraphicsPipeline(pipelineDesc);
        return pipelines;
    }

    void LightingRenderer::recordCompose(RenderWorker *worker, uint32_t sceneIndex) {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        if (scene.rect.isEmpty() || (sceneIndex >= composeSets.size())) {
            return;
        }

        RenderTarget *colorTarget = scene.colorTarget;
        ComposePipelines &pipelines = getComposePipelines(colorTarget->multisampling, colorTarget->format);
        colorTarget->setupColorFramebuffer(worker);

        interop::LightingComposeCB composeCB;
        composeCB.sceneIndex = sceneIndex;
        composeCB.padding = { 0, 0, 0 };
        worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(colorTarget->width), float(colorTarget->height)));
        worker->commandList->setScissors(scene.rect);
        worker->commandList->setPipeline((scene.params.settings.x != 0) ? pipelines.copy.get() : pipelines.multiply.get());
        worker->commandList->setGraphicsPipelineLayout(composePipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(composeSets[sceneIndex]->get(), 0);
        worker->commandList->setGraphicsPushConstants(0, &composeCB);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);
    }

    bool LightingRenderer::empty() const {
        return scenes.empty();
    }
};
