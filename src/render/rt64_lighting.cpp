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
#include "shaders/LightingAOCS.hlsl.spirv.h"
#include "shaders/LightingAOCSMS.hlsl.spirv.h"
#include "shaders/LightingAOBlurCS.hlsl.spirv.h"
#include "shaders/LightingEmissiveCS.hlsl.spirv.h"
#include "shaders/LightingEmissiveCSMS.hlsl.spirv.h"
#include "shaders/LightingEmissiveBlurCS.hlsl.spirv.h"
#include "shaders/LightingGBufferVS.hlsl.spirv.h"
#include "shaders/LightingGBufferPS.hlsl.spirv.h"
#include "shaders/LightingGBufferPSMS.hlsl.spirv.h"
#include "shaders/LightingGBufferFoliagePS.hlsl.spirv.h"
#include "shaders/LightingGBufferFoliagePSMS.hlsl.spirv.h"
#include "shaders/LightingShadowPS.hlsl.spirv.h"
#include "shaders/LightingShadowMergedPS.hlsl.spirv.h"
#include "shaders/LightingGBufferMergedPS.hlsl.spirv.h"
#include "shaders/LightingGBufferMergedPSMS.hlsl.spirv.h"
#include "shaders/LightingShadowVS.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/FullScreenVS.hlsl.dxil.h"
#   include "shaders/LightingCopyPS.hlsl.dxil.h"
#   include "shaders/LightingCopyPSMS.hlsl.dxil.h"
#   include "shaders/LightingComposePS.hlsl.dxil.h"
#   include "shaders/LightingComposePSMS.hlsl.dxil.h"
#   include "shaders/LightingAOCS.hlsl.dxil.h"
#   include "shaders/LightingAOCSMS.hlsl.dxil.h"
#   include "shaders/LightingAOBlurCS.hlsl.dxil.h"
#   include "shaders/LightingEmissiveCS.hlsl.dxil.h"
#   include "shaders/LightingEmissiveCSMS.hlsl.dxil.h"
#   include "shaders/LightingEmissiveBlurCS.hlsl.dxil.h"
#   include "shaders/LightingGBufferVS.hlsl.dxil.h"
#   include "shaders/LightingGBufferPS.hlsl.dxil.h"
#   include "shaders/LightingGBufferPSMS.hlsl.dxil.h"
#   include "shaders/LightingGBufferFoliagePS.hlsl.dxil.h"
#   include "shaders/LightingGBufferFoliagePSMS.hlsl.dxil.h"
#   include "shaders/LightingShadowPS.hlsl.dxil.h"
#   include "shaders/LightingShadowMergedPS.hlsl.dxil.h"
#   include "shaders/LightingGBufferMergedPS.hlsl.dxil.h"
#   include "shaders/LightingGBufferMergedPSMS.hlsl.dxil.h"
#   include "shaders/LightingShadowVS.hlsl.dxil.h"
#elif defined(__APPLE__)
#   include "shaders/FullScreenVS.hlsl.metal.h"
#   include "shaders/LightingCopyPS.hlsl.metal.h"
#   include "shaders/LightingCopyPSMS.hlsl.metal.h"
#   include "shaders/LightingComposePS.hlsl.metal.h"
#   include "shaders/LightingComposePSMS.hlsl.metal.h"
#   include "shaders/LightingAOCS.hlsl.metal.h"
#   include "shaders/LightingAOCSMS.hlsl.metal.h"
#   include "shaders/LightingAOBlurCS.hlsl.metal.h"
#   include "shaders/LightingEmissiveCS.hlsl.metal.h"
#   include "shaders/LightingEmissiveCSMS.hlsl.metal.h"
#   include "shaders/LightingEmissiveBlurCS.hlsl.metal.h"
#   include "shaders/LightingGBufferVS.hlsl.metal.h"
#   include "shaders/LightingGBufferPS.hlsl.metal.h"
#   include "shaders/LightingGBufferPSMS.hlsl.metal.h"
#   include "shaders/LightingGBufferFoliagePS.hlsl.metal.h"
#   include "shaders/LightingGBufferFoliagePSMS.hlsl.metal.h"
#   include "shaders/LightingShadowPS.hlsl.metal.h"
#   include "shaders/LightingShadowMergedPS.hlsl.metal.h"
#   include "shaders/LightingGBufferMergedPS.hlsl.metal.h"
#   include "shaders/LightingGBufferMergedPSMS.hlsl.metal.h"
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
        // RT64_LIGHTING forces the lighting on or off for tests regardless of the host's settings.
        if (getenv("RT64_LIGHTING") == nullptr) {
            rasterLightingEnabled = enabled;
        }
    }

    bool isRasterLightingEnabled() {
        return rasterLightingEnabled;
    }

    void setRasterLightingQuality(int quality) {
        rasterLightingQuality = std::clamp(quality, 0, 3);
    }

    int getRasterLightingQuality() {
        // RT64_LIGHT_QUALITY overrides the host's setting to compare the presets.
        const int overrideQuality = int(enhancementValue("RT64_LIGHT_QUALITY", -1.0f));
        return (overrideQuality >= 0) ? std::min(overrideQuality, 3) : int(rasterLightingQuality);
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

    // World space to the clip space of one face of a cube around a point light (90 degree field of view, depth = a + b / z
    // where z is the distance along the face's axis), as a row vector matrix. The faces and their axes are the ones of
    // lightingPointShadowFace in LightingComposePS.hlsl.
    static void computePointShadowFace(uint32_t face, const hlslpp::float3 &lightPosition, float zNear, float zFar, interop::float4x4 &matrix) {
        static const float Forward[6][3] = { { 1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -1.0f } };
        static const float Up[6][3] = { { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } };
        const hlslpp::float3 forward(Forward[face][0], Forward[face][1], Forward[face][2]);
        const hlslpp::float3 up(Up[face][0], Up[face][1], Up[face][2]);
        const hlslpp::float3 right = hlslpp::cross(up, forward);
        const float a = zFar / (zFar - zNear);
        const float b = -zFar * zNear / (zFar - zNear);
        hlslpp::float4x4 faceMatrix(
            float(right.x), float(up.x), a * float(forward.x), float(forward.x),
            float(right.y), float(up.y), a * float(forward.y), float(forward.y),
            float(right.z), float(up.z), a * float(forward.z), float(forward.z),
            -dot3(lightPosition, right), -dot3(lightPosition, up), -a * dot3(lightPosition, forward) + b, -dot3(lightPosition, forward));
        matrix = faceMatrix;
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

            // The faces of the point light's cube clip what's closer than their near plane instead (the player around the
            // origin of the shadows must not cover them).
            pipelineDesc.depthClipEnabled = true;
            pipelineDesc.pixelShader = nullptr;
            shadowOpaquePointPipeline = device->createGraphicsPipeline(pipelineDesc);
            pipelineDesc.pixelShader = pixelShader.get();
            shadowAlphaPointPipeline = device->createGraphicsPipeline(pipelineDesc);
            pipelineDesc.depthClipEnabled = false;

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

            // Where two surfaces both match the depth buffer within the tolerance (where they cross), the nearest one
            // keeps its normal instead of whichever was drawn last.
            gbufferDesc.depthEnabled = true;
            gbufferDesc.depthWriteEnabled = true;
            gbufferDesc.depthFunction = RenderComparisonFunction::LESS_EQUAL;
            gbufferDesc.depthTargetFormat = RenderFormat::D32_FLOAT;
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

            // Variants that draw consecutive draw calls together, finding the parameters of each triangle in a buffer
            // with the index of the primitive (which also needs geometry shader support). They bind five descriptor
            // sets: drivers that only take four (Quest 2) corrupt their stack when creating the layout.
            if (foliageNormalsSupported && (device->getCapabilities().maxDescriptorSets >= 5)) {
                LightingTriangleDrawSet triangleDrawSetDesc;
                layoutBuilder.begin(false, true);
                layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingShadowCB), RenderShaderStageFlag::VERTEX | RenderShaderStageFlag::PIXEL);
                layoutBuilder.addDescriptorSet(descriptorCommonSet);
                layoutBuilder.addDescriptorSet(descriptorTextureSet);
                layoutBuilder.addDescriptorSet(descriptorTextureSet);
                layoutBuilder.addDescriptorSet(descriptorFramebufferSet);
                layoutBuilder.addDescriptorSet(triangleDrawSetDesc);
                layoutBuilder.end();
                shadowMergedPipelineLayout = layoutBuilder.create(device);

                std::unique_ptr<RenderShader> shadowMergedPixelShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingShadowMergedPS, "PSMain", shaderFormat));
                pipelineDesc.pipelineLayout = shadowMergedPipelineLayout.get();
                pipelineDesc.pixelShader = shadowMergedPixelShader.get();
                shadowMergedPipeline = device->createGraphicsPipeline(pipelineDesc);
                pipelineDesc.depthClipEnabled = true;
                shadowMergedPointPipeline = device->createGraphicsPipeline(pipelineDesc);
                pipelineDesc.depthClipEnabled = false;

                layoutBuilder.begin(false, true);
                layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingGBufferCB), RenderShaderStageFlag::VERTEX | RenderShaderStageFlag::PIXEL);
                layoutBuilder.addDescriptorSet(descriptorCommonSet);
                layoutBuilder.addDescriptorSet(descriptorTextureSet);
                layoutBuilder.addDescriptorSet(descriptorTextureSet);
                layoutBuilder.addDescriptorSet(descriptorFramebufferSet);
                layoutBuilder.addDescriptorSet(triangleDrawSetDesc);
                layoutBuilder.end();
                gbufferMergedPipelineLayout = layoutBuilder.create(device);

                std::unique_ptr<RenderShader> gbufferMergedPixelShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingGBufferMergedPS, "PSMain", shaderFormat));
                std::unique_ptr<RenderShader> gbufferMergedPixelShaderMS = device->createShader(LIGHTING_SHADER_INPUTS(LightingGBufferMergedPSMS, "PSMain", shaderFormat));
                gbufferDesc.pipelineLayout = gbufferMergedPipelineLayout.get();
                gbufferDesc.pixelShader = gbufferMergedPixelShader.get();
                gbufferMergedPipelines[0] = device->createGraphicsPipeline(gbufferDesc);
                gbufferDesc.pixelShader = gbufferMergedPixelShaderMS.get();
                gbufferMergedPipelines[1] = device->createGraphicsPipeline(gbufferDesc);
                triangleDrawSet = std::make_unique<LightingTriangleDrawSet>(device);
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

        // Ambient occlusion at half resolution and its blur.
        {
            LightingAODescriptorSet descriptorSet;
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingAOCB), RenderShaderStageFlag::COMPUTE);
            layoutBuilder.addDescriptorSet(descriptorSet);
            layoutBuilder.end();
            aoPipelineLayout = layoutBuilder.create(device);

            std::unique_ptr<RenderShader> aoShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingAOCS, "CSMain", shaderFormat));
            std::unique_ptr<RenderShader> aoShaderMS = device->createShader(LIGHTING_SHADER_INPUTS(LightingAOCSMS, "CSMain", shaderFormat));
            aoPipeline = device->createComputePipeline(RenderComputePipelineDesc(aoPipelineLayout.get(), aoShader.get(), 8, 8, 1));
            aoPipelineMS = device->createComputePipeline(RenderComputePipelineDesc(aoPipelineLayout.get(), aoShaderMS.get(), 8, 8, 1));

            LightingAOBlurDescriptorSet blurDescriptorSet;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingAOBlurCB), RenderShaderStageFlag::COMPUTE);
            layoutBuilder.addDescriptorSet(blurDescriptorSet);
            layoutBuilder.end();
            aoBlurPipelineLayout = layoutBuilder.create(device);

            std::unique_ptr<RenderShader> blurShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingAOBlurCS, "CSMain", shaderFormat));
            aoBlurPipeline = device->createComputePipeline(RenderComputePipelineDesc(aoBlurPipelineLayout.get(), blurShader.get(), 8, 8, 1));
            aoBlurSets[0] = std::make_unique<LightingAOBlurDescriptorSet>(device);
            aoBlurSets[1] = std::make_unique<LightingAOBlurDescriptorSet>(device);
        }

        // Light of the glowing surfaces at quarter resolution and its blur.
        {
            LightingEmissiveDescriptorSet descriptorSet;
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingEmissiveCB), RenderShaderStageFlag::COMPUTE);
            layoutBuilder.addDescriptorSet(descriptorSet);
            layoutBuilder.end();
            emissivePipelineLayout = layoutBuilder.create(device);

            std::unique_ptr<RenderShader> emissiveShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingEmissiveCS, "CSMain", shaderFormat));
            std::unique_ptr<RenderShader> emissiveShaderMS = device->createShader(LIGHTING_SHADER_INPUTS(LightingEmissiveCSMS, "CSMain", shaderFormat));
            emissivePipeline = device->createComputePipeline(RenderComputePipelineDesc(emissivePipelineLayout.get(), emissiveShader.get(), 8, 8, 1));
            emissivePipelineMS = device->createComputePipeline(RenderComputePipelineDesc(emissivePipelineLayout.get(), emissiveShaderMS.get(), 8, 8, 1));

            LightingAOBlurDescriptorSet blurDescriptorSet;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingEmissiveBlurCB), RenderShaderStageFlag::COMPUTE);
            layoutBuilder.addDescriptorSet(blurDescriptorSet);
            layoutBuilder.end();
            emissiveBlurPipelineLayout = layoutBuilder.create(device);

            std::unique_ptr<RenderShader> blurShader = device->createShader(LIGHTING_SHADER_INPUTS(LightingEmissiveBlurCS, "CSMain", shaderFormat));
            emissiveBlurPipeline = device->createComputePipeline(RenderComputePipelineDesc(emissiveBlurPipelineLayout.get(), blurShader.get(), 8, 8, 1));
            emissiveBlurSets[0] = std::make_unique<LightingAOBlurDescriptorSet>(device);
            emissiveBlurSets[1] = std::make_unique<LightingAOBlurDescriptorSet>(device);
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
        if ((colorCopyTexture == nullptr) || (colorCopyFormat != colorTarget->format)) {
            // Scenes of the same frame with a different color format are rare; they skip the passes that need the copy.
            return nullptr;
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

        // Every copy of the frame gets its own descriptor set: updating one that an earlier copy of the same frame already
        // bound would invalidate the command list on Vulkan.
        while (copySets.size() <= copySetCursor) {
            copySets.emplace_back(std::make_unique<LightingCopyDescriptorSet>(device));
        }

        LightingCopyDescriptorSet *copySet = copySets[copySetCursor++].get();
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
        skyDesc.sceneIndex = sceneIndex;
        if (skyDesc.sceneColor != nullptr) {
            sky->record(worker, skyDesc);
        }
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
        if (postDesc.sceneColor != nullptr) {
            postEffects->record(worker, postDesc);
        }
    }

    LightingRenderer::~LightingRenderer() { }

    void LightingRenderer::reset() {
        gbufferEnabled = (enhancementValue("RT64_LIGHT_GBUFFER", 1.0f) > 0.0f);
        mergedDraws = (shadowMergedPipeline != nullptr) && (enhancementValue("RT64_LIGHT_MERGE_DRAWS", 1.0f) > 0.0f);
        bumpEnabled = (getRasterLightingQuality() >= 1) && (enhancementValue("RT64_LIGHT_BUMP", 0.0f) > 0.0f);
        scenes.clear();
        casters.clear();
        paramsVector.clear();
        shadowMapActive = false;
        shadowMapRendered = false;
        pointShadowActive = false;
        pointShadowRendered = false;
        copySetCursor = 0;
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

        // The lights and the focus were placed in the space of the game frame's geometry; a frame interpolated between two
        // game frames draws it elsewhere (the camera is applied to it), so they're moved along through the world.
        auto toDrawnFrame = [&](const hlslpp::float3 &v, bool position) {
            if (!desc.worldInterpolated) {
                return v;
            }

            const hlslpp::float3 relative = position ? (v - desc.gameWorldOrigin.xyz) : v;
            const hlslpp::float3 world(dot3(relative, desc.gameWorldRight), dot3(relative, desc.gameWorldUp), dot3(relative, desc.gameWorldForward));
            const hlslpp::float3 drawn = desc.worldRight * float(world.x) + desc.worldUp * float(world.y) + desc.worldForward * float(world.z);
            return position ? (drawn + desc.worldOrigin.xyz) : drawn;
        };

        scene.focusPosition = (float(desc.focusPosition.w) > 0.0f) ? hlslpp::float4(toDrawnFrame(desc.focusPosition.xyz, true), 1.0f) : desc.focusPosition;
        params.worldRight = hlslpp::float4(scene.worldRight, 0.0f);
        params.worldUp = hlslpp::float4(scene.worldUp, 0.0f);
        params.worldForward = hlslpp::float4(scene.worldForward, 0.0f);
        params.worldOrigin = hlslpp::float4(scene.worldOrigin, (float(desc.worldOrigin.w) > 0.0f) ? 1.0f : 0.0f);

        // The sun is the light placed much further away than anything else. Other lights are carried with the camera.
        const float sunStrength = enhancementValue("RT64_LIGHT_SUN", 0.9f);
        for (uint32_t i = 0; i < desc.lightCount; i++) {
            const interop::PointLight &light = desc.lights[i];
            const hlslpp::float3 gamePosition(light.position.x, light.position.y, light.position.z);
            const float distance = length3(gamePosition);
            const hlslpp::float3 position = toDrawnFrame(gamePosition, distance <= 1e6f);
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
                // A pool of warm light around the player that fades out faster than the path tracer's, so the rooms get
                // darker away from it instead of being brightened evenly.
                const float pointStrength = enhancementValue("RT64_LIGHT_POINT", 0.9f);
                // Development: the light can be moved along the world's axes to look at its shadows from a camera it sits on.
                const hlslpp::float3 debugOffset = scene.worldRight * enhancementValue("RT64_LIGHT_POINT_OFFSET_X", 0.0f) +
                    scene.worldUp * enhancementValue("RT64_LIGHT_POINT_OFFSET_Y", 0.0f) + scene.worldForward * enhancementValue("RT64_LIGHT_POINT_OFFSET_Z", 0.0f);
                params.pointLightPosition = hlslpp::float4(position + debugOffset, light.attenuationRadius * enhancementValue("RT64_LIGHT_POINT_RADIUS", 0.75f));
                params.pointLightColor = hlslpp::float4(light.diffuseColor.x * pointStrength, light.diffuseColor.y * pointStrength, light.diffuseColor.z * pointStrength,
                    enhancementValue("RT64_LIGHT_POINT_FALLOFF", 2.0f));
            }
        }

        if (scene.hasSun) {
            params.ambientColor = hlslpp::float4(enhancementValue("RT64_LIGHT_SKY_R", 0.62f), enhancementValue("RT64_LIGHT_SKY_G", 0.66f), enhancementValue("RT64_LIGHT_SKY_B", 0.74f), 0.0f);
            params.groundColor = hlslpp::float4(enhancementValue("RT64_LIGHT_GROUND_R", 0.50f), enhancementValue("RT64_LIGHT_GROUND_G", 0.47f), enhancementValue("RT64_LIGHT_GROUND_B", 0.42f), 0.0f);
        }
        else {
            // An even, almost neutral ambient: interiors have no light of their own by default (see RT64_RT_INDOOR_LIGHT),
            // so the ambient occlusion and the contact shadows give them their depth.
            const float indoorAmbient = enhancementValue("RT64_LIGHT_INDOOR_AMBIENT", 0.9f);
            const hlslpp::float3 indoorTint(enhancementValue("RT64_LIGHT_INDOOR_TINT_R", 0.97f), enhancementValue("RT64_LIGHT_INDOOR_TINT_G", 1.0f), enhancementValue("RT64_LIGHT_INDOOR_TINT_B", 1.03f));
            params.ambientColor = hlslpp::float4(indoorTint * indoorAmbient, 0.0f);
            const float indoorGround = indoorAmbient * enhancementValue("RT64_LIGHT_INDOOR_GROUND", 0.85f);
            params.groundColor = hlslpp::float4(indoorTint * indoorGround, 0.0f);
        }

        params.fog = hlslpp::float4(desc.fogMul, desc.fogOffset, desc.fogEnabled ? 1.0f : 0.0f, 0.0f);
        params.lightingParams = hlslpp::float4(enhancementValue("RT64_LIGHT_STRENGTH", 1.0f), enhancementValue("RT64_LIGHT_EXPOSURE", 1.0f), enhancementValue("RT64_LIGHT_WRAP", 0.5f), enhancementValue("RT64_LIGHT_SHADING", 1.0f));
        params.foliageParams = hlslpp::float4(enhancementValue("RT64_LIGHT_FOLIAGE_WRAP", 0.8f), enhancementValue("RT64_LIGHT_FOLIAGE_TRANSLUCENCY", 0.6f), enhancementValue("RT64_LIGHT_FOLIAGE_SHADOW", 0.35f), enhancementValue("RT64_LIGHT_FOLIAGE_SHADOW_OFFSET", 1.0f));
        static const float AOSlices[] = { 0.0f, 2.0f, 3.0f, 4.0f };
        params.aoParams = hlslpp::float4(enhancementValue("RT64_LIGHT_AO_RADIUS", 120.0f), enhancementValue("RT64_LIGHT_AO_STRENGTH", 0.8f), enhancementValue("RT64_LIGHT_AO_POWER", 1.3f), enhancementValue("RT64_LIGHT_AO_SLICES", AOSlices[getRasterLightingQuality()]));
        params.aoParams2 = hlslpp::float4(enhancementValue("RT64_LIGHT_AO_DIRECT", 0.35f), enhancementValue("RT64_LIGHT_AO_FOLIAGE", 0.3f), 0.0f, 0.0f);
        if (float(params.aoParams.y) <= 0.0f) {
            params.aoParams.w = 0.0f;
        }

        static const float ContactSteps[] = { 0.0f, 8.0f, 12.0f, 16.0f };
        const float contactStrength = enhancementValue("RT64_LIGHT_CONTACT_STRENGTH", 1.0f);
        params.contactParams = hlslpp::float4(enhancementValue("RT64_LIGHT_CONTACT_LENGTH", 120.0f), enhancementValue("RT64_LIGHT_CONTACT_THICKNESS", 30.0f), contactStrength,
            (contactStrength > 0.0f) ? enhancementValue("RT64_LIGHT_CONTACT_STEPS", ContactSteps[getRasterLightingQuality()]) : 0.0f);
        params.settings.x = uint32_t(enhancementValue("RT64_LIGHT_DEBUG", 0.0f));
        params.settings.y = uint32_t(getRasterLightingQuality());
        params.settings.z = (depthTarget != nullptr) ? depthTarget->multisampling.sampleCount : 1;
        params.settings.w = desc.skyHidden ? LIGHTING_SCENE_FLAG_SKY_HIDDEN : 0U;

        // Exteriors whose game sky isn't a daytime sky tint the light, from the analysis the procedural sky made of it on
        // the previous frames. Interiors, and scenes whose sky hasn't been analyzed, don't read it.
        const uint32_t sceneIndex = uint32_t(scenes.size());
        const float skyTint = enhancementValue("RT64_LIGHT_SKY_TINT", 0.8f);
        if (scene.hasSun && !desc.skyHidden && (skyTint > 0.0f) && sky->enabled() && sky->isAnalysisValid(sceneIndex)) {
            params.settings.w |= LIGHTING_SCENE_FLAG_SKY_TINT;
        }

        params.miscParams = hlslpp::float4(skyTint, enhancementValue("RT64_LIGHT_AO_MIN_HEIGHT", 12.0f), 0.0f, 0.0f);

        // The clouds of the procedural sky shadow the ground in exteriors whose sky it replaces (the shader weighs them by
        // how much of the game's sky it replaces, from the analysis).
        interop::float4 cloudShadowParams(0.0f, 0.0f, 0.0f, 0.0f), cloudShadowOffset(0.0f, 0.0f, 0.0f, 0.0f), cloudShadowMisc(0.0f, 0.0f, 0.0f, 0.0f);
        if (scene.hasSun && sky->enabled() && (desc.skyHidden || sky->isAnalysisValid(sceneIndex))) {
            sky->getCloudShadow(lightingTime(), getRasterLightingQuality(), cloudShadowParams, cloudShadowOffset, cloudShadowMisc);
        }

        params.cloudShadowParams = hlslpp::float4(cloudShadowParams.x, cloudShadowParams.y, cloudShadowParams.z, cloudShadowParams.w);
        params.cloudShadowOffset = hlslpp::float4(cloudShadowOffset.x, cloudShadowOffset.y, cloudShadowOffset.z, cloudShadowOffset.w);
        params.cloudShadowMisc = hlslpp::float4(cloudShadowMisc.x, cloudShadowMisc.y, cloudShadowMisc.z, cloudShadowMisc.w);

        // Glowing surfaces in scenes without a sun (interiors and dungeons). Experimental and off by default: they cost
        // about 0.2 ms (a copy of the color target and three small passes), and in Mega Man 64 the colors alone can't
        // tell its few small lamps from bright banners and signs (RT64_LIGHT_EMISSIVE_MAX 0.6 turns them on).
        const float emissiveMax = enhancementValue("RT64_LIGHT_EMISSIVE_MAX", 0.0f);
        if (!scene.hasSun && (emissiveMax > 0.0f) && (getRasterLightingQuality() >= int(enhancementValue("RT64_LIGHT_EMISSIVE_QUALITY", 2.0f)))) {
            params.emissiveParams = hlslpp::float4(enhancementValue("RT64_LIGHT_EMISSIVE_THRESHOLD", 0.65f), enhancementValue("RT64_LIGHT_EMISSIVE", 0.35f),
                enhancementValue("RT64_LIGHT_EMISSIVE_LIGHT", 10.0f), emissiveMax);
        }
        else {
            params.emissiveParams = hlslpp::float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        if (gbufferEnabled) {
            params.settings.w |= LIGHTING_SCENE_FLAG_GBUFFER;
        }

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

    void LightingRenderer::addCaster(uint32_t sceneIndex, uint32_t instanceIndex, bool alphaTested) {
        casters.push_back({ sceneIndex, instanceIndex, alphaTested });
    }

    void LightingRenderer::addGBufferDraw(uint32_t sceneIndex, uint32_t instanceIndex, uint32_t flags) {
        assert(sceneIndex < scenes.size());
        if (!gbufferEnabled) {
            return;
        }

        // Relief from the textures costs a few extra samples per pixel, so it's left out of the low quality preset.
        if (!bumpEnabled) {
            flags &= ~LIGHTING_GBUFFER_BUMP;
        }

        if (!foliageNormalsSupported) {
            flags &= ~LIGHTING_GBUFFER_FOLIAGE;
        }

        scenes[sceneIndex].gbufferDraws.push_back({ instanceIndex, flags });
    }

    void LightingRenderer::createAOTextures(uint32_t width, uint32_t height) {
        for (uint32_t i = 0; i < 2; i++) {
            aoTextures[i] = device->createTexture(RenderTextureDesc::Texture2D(width, height, 1, RenderFormat::R16G16B16A16_FLOAT, RenderTextureFlag::STORAGE | RenderTextureFlag::UNORDERED_ACCESS));
            aoTextures[i]->setName("Lighting Ambient Occlusion");
        }

        aoTextureWidth = width;
        aoTextureHeight = height;
    }

    void LightingRenderer::createNormalBuffer(RenderWorker *worker, uint32_t width, uint32_t height) {
        normalFramebuffer.reset();
        normalBuffer.reset();
        normalDepth.reset();
        normalBuffer = device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_UNORM));
        normalBuffer->setName("Lighting Normal Buffer");
        normalDepth = device->createTexture(RenderTextureDesc::DepthTarget(width, height, RenderFormat::D32_FLOAT));
        normalDepth->setName("Lighting Normal Buffer Depth");
        const RenderTexture *colorAttachment = normalBuffer.get();
        normalFramebuffer = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1, normalDepth.get()));
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

    void LightingRenderer::createPointShadowMap(uint32_t faceSize) {
        pointShadowFramebuffer.reset();
        pointShadowMapView.reset();
        pointShadowMap.reset();
        pointShadowMap = device->createTexture(RenderTextureDesc::DepthTarget(faceSize * 3, faceSize * 2, RenderFormat::D32_FLOAT));
        pointShadowMap->setName("Lighting Point Shadow Map");
        pointShadowMapView = pointShadowMap->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::D32_FLOAT));
        pointShadowFramebuffer = device->createFramebuffer(RenderFramebufferDesc(nullptr, 0, pointShadowMap.get()));
        pointShadowFaceSize = faceSize;
        pointShadowNeedsTransition = true;
    }

    void LightingRenderer::finish(RenderWorker *worker, const std::vector<InstanceDrawCall> &instanceDrawCalls, std::vector<BufferUploader::Upload> &uploads) {
        if (scenes.empty()) {
            return;
        }

        // The shadow map follows the biggest scene with a sun (the main view; the first one of equal size, like the left
        // eye in VR), and only its draw calls cast shadows in it. Other scenes are often the same world seen from elsewhere (the other eye in VR), which would draw every
        // caster twice.
        uint32_t sunSceneIndex = UINT32_MAX;
        int64_t sunSceneArea = -1;
        for (uint32_t i = 0; i < uint32_t(scenes.size()); i++) {
            const RenderRect &rect = scenes[i].rect;
            const int64_t area = rect.isEmpty() ? 0 : (int64_t(rect.right - rect.left) * int64_t(rect.bottom - rect.top));
            if (scenes[i].hasSun && (area > sunSceneArea)) {
                sunSceneIndex = i;
                sunSceneArea = area;
            }
        }

        // Without a sun, the light carried in the biggest scene casts shadows in all directions instead. They're drawn from
        // around the chest of the player instead of from the light itself, which floats high above them and can end up
        // above a low ceiling that would shadow the whole room; without the player's position there are none (the light
        // would be placed from the camera alone). Development: RT64_LIGHT_POINT_SHADOW_ORIGIN 1 draws them from the light.
        uint32_t pointSceneIndex = UINT32_MAX;
        hlslpp::float3 pointShadowOrigin(0.0f, 0.0f, 0.0f);
        if (sunSceneIndex == UINT32_MAX) {
            const bool fromLight = (enhancementValue("RT64_LIGHT_POINT_SHADOW_ORIGIN", 0.0f) > 0.0f);
            const float originHeight = enhancementValue("RT64_LIGHT_POINT_SHADOW_HEIGHT", 110.0f);
            int64_t pointSceneArea = -1;
            for (uint32_t i = 0; i < uint32_t(scenes.size()); i++) {
                const Scene &scene = scenes[i];
                const RenderRect &rect = scene.rect;
                const int64_t area = rect.isEmpty() ? 0 : (int64_t(rect.right - rect.left) * int64_t(rect.bottom - rect.top));
                const interop::float4 &light = scene.params.pointLightPosition;
                const float maxFocusDistance = enhancementValue("RT64_RT_INDOOR_FOCUS_MAX_DISTANCE", 2500.0f);
                const bool focusKnown = (float(scene.focusPosition.w) > 0.0f) && (length3(scene.focusPosition.xyz - scene.cameraPosition) < maxFocusDistance);
                if ((float(light.w) <= 0.0f) || !(focusKnown || fromLight) || (area <= pointSceneArea)) {
                    continue;
                }

                pointSceneIndex = i;
                pointSceneArea = area;
                pointShadowOrigin = fromLight ? hlslpp::float3(light.x, light.y, light.z) : (scene.focusPosition.xyz + scene.worldUp * originHeight);
            }
        }

        const uint32_t casterSceneIndex = (sunSceneIndex != UINT32_MAX) ? sunSceneIndex : pointSceneIndex;
        casters.erase(std::remove_if(casters.begin(), casters.end(), [&](const Caster &caster) { return caster.sceneIndex != casterSceneIndex; }), casters.end());

        // Parameters of each triangle of the draw calls drawn again by the shadow and normal passes.
        if (mergedDraws) {
            uint32_t triangleCount = 1;
            auto updateTriangleCount = [&](uint32_t instanceIndex) {
                const InstanceDrawCall &drawCall = instanceDrawCalls[instanceIndex];
                if (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) {
                    triangleCount = std::max(triangleCount, drawCall.triangles.indexStart / 3 + drawCall.triangles.faceCount);
                }
            };

            for (const Caster &caster : casters) {
                updateTriangleCount(caster.instanceIndex);
            }

            for (const Scene &scene : scenes) {
                for (const GBufferDraw &draw : scene.gbufferDraws) {
                    updateTriangleCount(draw.instanceIndex);
                }
            }

            triangleDraws.assign(triangleCount, 0);
            auto writeTriangles = [&](uint32_t instanceIndex, uint32_t flags) {
                const InstanceDrawCall &drawCall = instanceDrawCalls[instanceIndex];
                if (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) {
                    const uint32_t firstTriangle = drawCall.triangles.indexStart / 3;
                    const uint32_t value = (instanceIndex & 0xFFFFFFU) | (flags << 24);
                    std::fill(triangleDraws.begin() + firstTriangle, triangleDraws.begin() + firstTriangle + drawCall.triangles.faceCount, value);
                }
            };

            for (const Caster &caster : casters) {
                writeTriangles(caster.instanceIndex, caster.alphaTested ? LIGHTING_GBUFFER_ALPHA_TESTED : 0);
            }

            // The flags of the normal pass include the ones of the shadow pass.
            for (const Scene &scene : scenes) {
                for (const GBufferDraw &draw : scene.gbufferDraws) {
                    writeTriangles(draw.instanceIndex, draw.flags);
                }
            }

            uploads.push_back({ triangleDraws.data(), { 0, triangleDraws.size() }, sizeof(uint32_t), RenderBufferFlag::STORAGE, { }, &triangleDrawsBuffer });
        }

        static const uint32_t ShadowMapSizes[] = { 1024, 2048, 2048, 4096 };
        const uint32_t desiredSize = uint32_t(enhancementValue("RT64_LIGHT_SHADOW_SIZE", float(ShadowMapSizes[getRasterLightingQuality()])));
        const uint32_t mapSize = std::clamp(desiredSize, 256U, 8192U);
        if ((shadowMap == nullptr) || (shadowMapSize != mapSize)) {
            createShadowMap(worker, mapSize);
        }

        // The point light's map always exists (the composition reads it), at a tiny size until it's needed.
        static const uint32_t PointShadowSizes[] = { 0, 512, 768, 1024 };
        const uint32_t pointFaceSize = uint32_t(std::clamp(enhancementValue("RT64_LIGHT_POINT_SHADOW_SIZE", float(PointShadowSizes[getRasterLightingQuality()])), 0.0f, 2048.0f));
        const float pointShadowStrength = enhancementValue("RT64_LIGHT_POINT_SHADOW", 1.0f);
        pointShadowActive = (pointSceneIndex != UINT32_MAX) && !casters.empty() && (pointFaceSize >= 64) && (pointShadowStrength > 0.0f);
        if ((pointShadowMap == nullptr) || (pointShadowActive && (pointShadowFaceSize != pointFaceSize))) {
            createPointShadowMap(pointShadowActive ? pointFaceSize : 16);
        }

        hlslpp::float3 pointShadowLight(0.0f, 0.0f, 0.0f);
        float pointShadowNear = 1.0f;
        float pointShadowFar = 2.0f;
        if (pointShadowActive) {
            const Scene &pointScene = scenes[pointSceneIndex];
            pointShadowLight = pointShadowOrigin;
            pointShadowFar = std::max(float(pointScene.params.pointLightPosition.w), 200.0f);
            pointShadowNear = std::clamp(enhancementValue("RT64_LIGHT_POINT_SHADOW_NEAR", 80.0f), 1.0f, pointShadowFar * 0.5f);
            for (uint32_t face = 0; face < 6; face++) {
                computePointShadowFace(face, pointShadowLight, pointShadowNear, pointShadowFar, pointShadowMatrices[face]);
            }
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

        const uint32_t aoWidth = std::max((normalWidth + 1) / 2, 1U);
        const uint32_t aoHeight = std::max((normalHeight + 1) / 2, 1U);
        if ((aoTextures[0] == nullptr) || (aoTextureWidth < aoWidth) || (aoTextureHeight < aoHeight)) {
            createAOTextures(std::max(aoWidth, aoTextureWidth), std::max(aoHeight, aoTextureHeight));
        }

        const uint32_t emissiveWidth = std::max((normalWidth + 3) / 4, 1U);
        const uint32_t emissiveHeight = std::max((normalHeight + 3) / 4, 1U);
        if ((emissiveTextures[0] == nullptr) || (emissiveTextureWidth < emissiveWidth) || (emissiveTextureHeight < emissiveHeight)) {
            emissiveTextureWidth = std::max(emissiveWidth, emissiveTextureWidth);
            emissiveTextureHeight = std::max(emissiveHeight, emissiveTextureHeight);
            for (uint32_t i = 0; i < 2; i++) {
                emissiveTextures[i] = device->createTexture(RenderTextureDesc::Texture2D(emissiveTextureWidth, emissiveTextureHeight, 1, RenderFormat::R16G16B16A16_FLOAT, RenderTextureFlag::STORAGE | RenderTextureFlag::UNORDERED_ACCESS));
                emissiveTextures[i]->setName("Lighting Emissive Light");
            }
        }

        frameIndex++;

        RenderFormat colorFormat = scenes[0].colorTarget->format;
        if ((colorCopyTexture == nullptr) || (colorCopyWidth < normalWidth) || (colorCopyHeight < normalHeight) || (colorCopyFormat != colorFormat)) {
            colorCopyFramebuffer.reset();
            colorCopyTexture.reset();
            colorCopyWidth = std::max(colorCopyWidth, normalWidth);
            colorCopyHeight = std::max(colorCopyHeight, normalHeight);
            colorCopyFormat = colorFormat;
            colorCopyTexture = device->createTexture(RenderTextureDesc::ColorTarget(colorCopyWidth, colorCopyHeight, colorCopyFormat));
            colorCopyTexture->setName("Lighting Color Copy");
            const RenderTexture *colorAttachment = colorCopyTexture.get();
            colorCopyFramebuffer = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1));
        }

        const Scene *sunScene = (sunSceneIndex < scenes.size()) ? &scenes[sunSceneIndex] : nullptr;

        const float shadowStrength = enhancementValue("RT64_LIGHT_SHADOW_STRENGTH", 1.0f);
        float texelSize = 1.0f;
        float depthRange = 1.0f;
        shadowMapActive = (sunScene != nullptr) && !casters.empty() && (shadowStrength > 0.0f);
        if (shadowMapActive) {
            // A square of a fixed size seen from the sun, around the player (or the camera when the host doesn't give the
            // player's position), so turning the camera never changes what the shadow map covers and its texels stay
            // fixed in the world while it follows. A fit around the view frustum moved with every turn of the camera:
            // shadows past its far end came and went, and on interpolated frames its texels crawled.
            const float radius = std::max(ceilf(enhancementValue("RT64_LIGHT_SHADOW_RADIUS", 3000.0f) / 16.0f) * 16.0f, 16.0f);
            const float casterDistance = enhancementValue("RT64_LIGHT_SHADOW_CASTER_DISTANCE", 6000.0f);
            const float maxFocusDistance = enhancementValue("RT64_RT_INDOOR_FOCUS_MAX_DISTANCE", 2500.0f);
            const bool focusKnown = (float(sunScene->focusPosition.w) > 0.0f) && (length3(sunScene->focusPosition.xyz - sunScene->cameraPosition) < maxFocusDistance);
            const hlslpp::float3 center = focusKnown ? sunScene->focusPosition.xyz : sunScene->cameraPosition;
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
                params.shadowMapParams = hlslpp::float4(1.0f / mapSize, 1.0f / mapSize, shadowStrength, depthRange);

                // From medium quality up the penumbra widens with the distance to the caster, like under a real sun.
                static const float MaxSoftness[] = { 0.0f, 5.0f, 6.0f, 8.0f };
                const int quality = getRasterLightingQuality();
                const float sunSize = (quality >= 1) ? enhancementValue("RT64_LIGHT_SHADOW_SUN_SIZE", 0.025f) : 0.0f;
                params.shadowParams2 = hlslpp::float4(tanf(std::clamp(sunSize, 0.0f, 0.3f)), enhancementValue("RT64_LIGHT_SHADOW_MIN_SOFTNESS", 0.75f),
                    enhancementValue("RT64_LIGHT_SHADOW_MAX_SOFTNESS", MaxSoftness[quality]), 0.0f);
            }
            else {
                params.shadowMapParams = hlslpp::float4(1.0f / mapSize, 1.0f / mapSize, 0.0f, 0.0f);
                params.shadowParams2 = hlslpp::float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            if (pointShadowActive && !scene.hasSun && (float(params.pointLightPosition.w) > 0.0f)) {
                const float a = pointShadowFar / (pointShadowFar - pointShadowNear);
                const float b = -pointShadowFar * pointShadowNear / (pointShadowFar - pointShadowNear);
                params.pointShadowParams = hlslpp::float4(a, b, float(pointShadowFaceSize), pointShadowStrength);
                params.pointShadowPosition = hlslpp::float4(pointShadowLight, pointShadowNear);
                params.pointShadowParams2 = hlslpp::float4(enhancementValue("RT64_LIGHT_POINT_SHADOW_NORMAL_OFFSET", 1.5f), enhancementValue("RT64_LIGHT_POINT_SHADOW_BIAS", 1.5f),
                    enhancementValue("RT64_LIGHT_POINT_SHADOW_SOFTNESS", 1.0f), 0.0f);
            }
            else {
                params.pointShadowParams = hlslpp::float4(0.0f, 0.0f, 0.0f, 0.0f);
                params.pointShadowPosition = hlslpp::float4(0.0f, 0.0f, 0.0f, 0.0f);
                params.pointShadowParams2 = hlslpp::float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            // Size of the region of the ambient occlusion texture used by the scene.
            const uint32_t sceneWidth = uint32_t(std::max(scene.rect.right - scene.rect.left, 0));
            const uint32_t sceneHeight = uint32_t(std::max(scene.rect.bottom - scene.rect.top, 0));
            params.aoParams2.z = float(std::min((sceneWidth + 1) / 2, aoTextureWidth));
            params.aoParams2.w = float(std::min((sceneHeight + 1) / 2, aoTextureHeight));

            // The glowing surfaces are found in the copy of the color target, which only holds one format per frame.
            if (scene.colorTarget->format != colorCopyFormat) {
                params.emissiveParams.w = 0.0f;
            }

            paramsVector.emplace_back(params);
        }

        uploads.push_back({ paramsVector.data(), { 0, paramsVector.size() }, sizeof(interop::LightingParams), RenderBufferFlag::STORAGE, { }, &paramsBuffer });

        static const bool printScenes = (getenv("RT64_LIGHT_PRINT") != nullptr);
        static uint32_t printCounter = 0;
        const uint32_t printInterval = std::max(uint32_t(enhancementValue("RT64_LIGHT_PRINT_INTERVAL", 120.0f)), 1U);
        if (printScenes && ((printCounter++ % printInterval) == 0)) {
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
            set->setTexture(set->gAmbientOcclusion, aoTextures[0].get(), RenderTextureLayout::SHADER_READ);
            set->setBuffer(set->gSkyAnalysis, sky->getAnalysisBuffer(), RenderBufferStructuredView(sizeof(interop::float4)));
            set->setTexture(set->gSceneColor, colorCopyTexture.get(), RenderTextureLayout::SHADER_READ);
            set->setTexture(set->gEmissiveLight, emissiveTextures[0].get(), RenderTextureLayout::SHADER_READ);
            set->setTexture(set->gPointShadowMap, pointShadowMap.get(), RenderTextureLayout::DEPTH_READ, pointShadowMapView.get());
        }

        while (emissiveSets.size() < scenes.size()) {
            emissiveSets.emplace_back(std::make_unique<LightingEmissiveDescriptorSet>(device));
        }

        for (uint32_t i = 0; i < uint32_t(scenes.size()); i++) {
            const Scene &scene = scenes[i];
            LightingEmissiveDescriptorSet *set = emissiveSets[i].get();
            set->setBuffer(set->gLightingParams, paramsBuffer.get(), RenderBufferStructuredView(sizeof(interop::LightingParams)));
            set->setTexture(set->gDepth, scene.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, scene.depthTarget->textureView.get());
            set->setTexture(set->gSceneColor, colorCopyTexture.get(), RenderTextureLayout::SHADER_READ);
            set->setTexture(set->gOutput, emissiveTextures[0].get(), RenderTextureLayout::GENERAL);
        }

        emissiveBlurSets[0]->setTexture(emissiveBlurSets[0]->gInput, emissiveTextures[0].get(), RenderTextureLayout::SHADER_READ);
        emissiveBlurSets[0]->setTexture(emissiveBlurSets[0]->gOutput, emissiveTextures[1].get(), RenderTextureLayout::GENERAL);
        emissiveBlurSets[1]->setTexture(emissiveBlurSets[1]->gInput, emissiveTextures[1].get(), RenderTextureLayout::SHADER_READ);
        emissiveBlurSets[1]->setTexture(emissiveBlurSets[1]->gOutput, emissiveTextures[0].get(), RenderTextureLayout::GENERAL);

        if (mergedDraws && (triangleDrawSet != nullptr) && (triangleDrawsBuffer.get() != nullptr)) {
            triangleDrawSet->setBuffer(triangleDrawSet->gTriangleDraws, triangleDrawsBuffer.get(), RenderBufferStructuredView(sizeof(uint32_t)));
        }

        while (aoSets.size() < scenes.size()) {
            aoSets.emplace_back(std::make_unique<LightingAODescriptorSet>(device));
        }

        for (uint32_t i = 0; i < uint32_t(scenes.size()); i++) {
            const Scene &scene = scenes[i];
            LightingAODescriptorSet *set = aoSets[i].get();
            set->setBuffer(set->gLightingParams, paramsBuffer.get(), RenderBufferStructuredView(sizeof(interop::LightingParams)));
            set->setTexture(set->gDepth, scene.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, scene.depthTarget->textureView.get());
            set->setTexture(set->gNormalBuffer, normalBuffer.get(), RenderTextureLayout::SHADER_READ);
            set->setTexture(set->gOutput, aoTextures[0].get(), RenderTextureLayout::GENERAL);
        }

        aoBlurSets[0]->setTexture(aoBlurSets[0]->gInput, aoTextures[0].get(), RenderTextureLayout::SHADER_READ);
        aoBlurSets[0]->setTexture(aoBlurSets[0]->gOutput, aoTextures[1].get(), RenderTextureLayout::GENERAL);
        aoBlurSets[1]->setTexture(aoBlurSets[1]->gInput, aoTextures[1].get(), RenderTextureLayout::SHADER_READ);
        aoBlurSets[1]->setTexture(aoBlurSets[1]->gOutput, aoTextures[0].get(), RenderTextureLayout::GENERAL);
    }

    void LightingRenderer::recordShadowMap(RenderWorker *worker, RenderDescriptorSet *commonSet, RenderDescriptorSet *textureSet, RenderDescriptorSet *framebufferSet,
        const RenderVertexBufferView *vertexViews, const RenderInputSlot *inputSlots, uint32_t vertexViewCount, const RenderIndexBufferView *indexView,
        const std::vector<InstanceDrawCall> &instanceDrawCalls)
    {
        if (shadowMap == nullptr) {
            return;
        }

        // The shadow maps must always be readable, even on frames that don't draw them.
        if (shadowMapNeedsTransition) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(shadowMap.get(), RenderTextureLayout::DEPTH_WRITE));
            worker->commandList->setFramebuffer(shadowFramebuffer.get());
            worker->commandList->clearDepth(true, 1.0f);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(shadowMap.get(), RenderTextureLayout::DEPTH_READ));
            shadowMapNeedsTransition = false;
        }

        if ((pointShadowMap != nullptr) && pointShadowNeedsTransition) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(pointShadowMap.get(), RenderTextureLayout::DEPTH_WRITE));
            worker->commandList->setFramebuffer(pointShadowFramebuffer.get());
            worker->commandList->clearDepth(true, 1.0f);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(pointShadowMap.get(), RenderTextureLayout::DEPTH_READ));
            pointShadowNeedsTransition = false;
        }

        const bool drawSun = shadowMapActive && !shadowMapRendered;
        const bool drawPoint = pointShadowActive && !pointShadowRendered && (pointShadowMap != nullptr);
        if (!drawSun && !drawPoint) {
            return;
        }

        RenderTexture *targetMap = drawSun ? shadowMap.get() : pointShadowMap.get();
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(targetMap, RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(drawSun ? shadowFramebuffer.get() : pointShadowFramebuffer.get());
        worker->commandList->clearDepth(true, 1.0f);
        worker->commandList->setVertexBuffers(0, vertexViews, vertexViewCount, inputSlots);
        worker->commandList->setIndexBuffer(indexView);

        interop::LightingShadowCB shadowCB;
        shadowCB.shadowMatrix = shadowMatrix;
        shadowCB.padding = { 0, 0, 0 };

        // Opaque casters don't need their draw call parameters, so consecutive ranges of indices are drawn together. They're
        // all drawn first so the depth test rejects as many pixels of the alpha tested casters as possible, which are
        // expensive as they sample their textures like the RDP does.
        const bool alphaTestedShadows = (enhancementValue("RT64_LIGHT_SHADOW_ALPHA", 1.0f) > 0.0f);
        if (!mergedDraws) {
            sortedCasters.clear();
            for (const Caster &caster : casters) {
                if (!caster.alphaTested || !alphaTestedShadows) {
                    sortedCasters.push_back({ caster.sceneIndex, caster.instanceIndex, false });
                }
            }

            if (alphaTestedShadows) {
                for (const Caster &caster : casters) {
                    if (caster.alphaTested) {
                        sortedCasters.push_back(caster);
                    }
                }
            }
        }

        if (mergedDraws) {
            worker->commandList->setGraphicsPipelineLayout(shadowMergedPipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(commonSet, 0);
            worker->commandList->setGraphicsDescriptorSet(textureSet, 1);
            worker->commandList->setGraphicsDescriptorSet(textureSet, 2);
            worker->commandList->setGraphicsDescriptorSet(framebufferSet, 3);
            worker->commandList->setGraphicsDescriptorSet(triangleDrawSet->get(), 4);
            worker->commandList->setPipeline(drawSun ? shadowMergedPipeline.get() : shadowMergedPointPipeline.get());
        }
        else {
            worker->commandList->setGraphicsPipelineLayout(shadowPipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(commonSet, 0);
            worker->commandList->setGraphicsDescriptorSet(textureSet, 1);
            worker->commandList->setGraphicsDescriptorSet(textureSet, 2);
            worker->commandList->setGraphicsDescriptorSet(framebufferSet, 3);
        }

        const RenderPipeline *opaquePipeline = drawSun ? shadowOpaquePipeline.get() : shadowOpaquePointPipeline.get();
        const RenderPipeline *alphaPipeline = drawSun ? shadowAlphaPipeline.get() : shadowAlphaPointPipeline.get();
        auto drawCasters = [&](const interop::float4x4 &matrix, const RenderViewport &viewport, const RenderRect &scissor) {
            worker->commandList->setViewports(viewport);
            worker->commandList->setScissors(scissor);
            shadowCB.shadowMatrix = matrix;

            // All the casters that follow each other in the index buffer are drawn at once.
            if (mergedDraws) {
                uint32_t runIndexStart = 0;
                uint32_t runIndexCount = 0;
                auto flushRun = [&]() {
                    if (runIndexCount > 0) {
                        shadowCB.renderIndex = runIndexStart / 3;
                        worker->commandList->setGraphicsPushConstants(0, &shadowCB);
                        worker->commandList->drawIndexedInstanced(runIndexCount, 1, runIndexStart, 0, 0);
                        runIndexCount = 0;
                    }
                };

                for (const Caster &caster : casters) {
                    const InstanceDrawCall &drawCall = instanceDrawCalls[caster.instanceIndex];
                    if ((drawCall.type != InstanceDrawCall::Type::IndexedTriangles) || (drawCall.triangles.faceCount == 0)) {
                        continue;
                    }

                    if ((runIndexCount == 0) || ((runIndexStart + runIndexCount) != drawCall.triangles.indexStart)) {
                        flushRun();
                        runIndexStart = drawCall.triangles.indexStart;
                    }

                    runIndexCount += drawCall.triangles.faceCount * 3;
                }

                flushRun();
                return;
            }

            const RenderPipeline *previousPipeline = nullptr;
            uint32_t pendingIndexStart = 0;
            uint32_t pendingIndexCount = 0;
            auto flushPending = [&]() {
                if (pendingIndexCount > 0) {
                    worker->commandList->drawIndexedInstanced(pendingIndexCount, 1, pendingIndexStart, 0, 0);
                    pendingIndexCount = 0;
                }
            };

            for (const Caster &caster : sortedCasters) {
                const InstanceDrawCall &drawCall = instanceDrawCalls[caster.instanceIndex];
                if ((drawCall.type != InstanceDrawCall::Type::IndexedTriangles) || (drawCall.triangles.faceCount == 0)) {
                    continue;
                }

                const RenderPipeline *pipeline = caster.alphaTested ? alphaPipeline : opaquePipeline;
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
        };

        if (drawSun) {
            drawCasters(shadowMatrix, RenderViewport(0.0f, 0.0f, float(shadowMapSize), float(shadowMapSize)), RenderRect(0, 0, int32_t(shadowMapSize), int32_t(shadowMapSize)));
            shadowMapRendered = true;
        }
        else {
            // The six faces of the cube in a 3x2 atlas (column = face % 3, row = face / 3).
            const int32_t size = int32_t(pointShadowFaceSize);
            for (uint32_t face = 0; face < 6; face++) {
                const int32_t x = int32_t(face % 3) * size;
                const int32_t y = int32_t(face / 3) * size;
                drawCasters(pointShadowMatrices[face], RenderViewport(float(x), float(y), float(size), float(size)), RenderRect(x, y, x + size, y + size));
            }

            pointShadowRendered = true;
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(targetMap, RenderTextureLayout::DEPTH_READ));
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

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
            RenderTextureBarrier(normalBuffer.get(), RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(normalDepth.get(), RenderTextureLayout::DEPTH_WRITE)
        });

        worker->commandList->setFramebuffer(normalFramebuffer.get());
        worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f), &scene.rect, 1);
        worker->commandList->clearDepth(true, 1.0f, &scene.rect, 1);
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
            gbufferCB.bumpStrength = enhancementValue("RT64_LIGHT_BUMP", 0.0f);
            if (mergedDraws) {
                worker->commandList->setGraphicsPipelineLayout(gbufferMergedPipelineLayout.get());
                worker->commandList->setGraphicsDescriptorSet(commonSet, 0);
                worker->commandList->setGraphicsDescriptorSet(textureSet, 1);
                worker->commandList->setGraphicsDescriptorSet(textureSet, 2);
                worker->commandList->setGraphicsDescriptorSet(framebufferSet, 3);
                worker->commandList->setGraphicsDescriptorSet(triangleDrawSet->get(), 4);
                worker->commandList->setPipeline(gbufferMergedPipelines[multisampling ? 1 : 0].get());
                uint32_t runIndexStart = 0;
                uint32_t runIndexCount = 0;
                auto flushRun = [&]() {
                    if (runIndexCount > 0) {
                        gbufferCB.renderIndex = runIndexStart / 3;
                        gbufferCB.indexStart = runIndexStart;
                        gbufferCB.flags = 0;
                        worker->commandList->setGraphicsPushConstants(0, &gbufferCB);
                        worker->commandList->drawIndexedInstanced(runIndexCount, 1, runIndexStart, 0, 0);
                        runIndexCount = 0;
                    }
                };

                for (const GBufferDraw &draw : scene.gbufferDraws) {
                    const InstanceDrawCall &drawCall = instanceDrawCalls[draw.instanceIndex];
                    if ((drawCall.type != InstanceDrawCall::Type::IndexedTriangles) || (drawCall.triangles.faceCount == 0)) {
                        continue;
                    }

                    if ((runIndexCount == 0) || ((runIndexStart + runIndexCount) != drawCall.triangles.indexStart)) {
                        flushRun();
                        runIndexStart = drawCall.triangles.indexStart;
                    }

                    runIndexCount += drawCall.triangles.faceCount * 3;
                }

                flushRun();
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(normalBuffer.get(), RenderTextureLayout::SHADER_READ));
                return;
            }

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

    void LightingRenderer::recordAmbientOcclusion(RenderWorker *worker, uint32_t sceneIndex) {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        const uint32_t width = uint32_t(scene.params.aoParams2.z);
        const uint32_t height = uint32_t(scene.params.aoParams2.w);
        const bool aoEnabled = (scene.params.aoParams.w > 0.0f);
        const bool contactEnabled = (scene.params.contactParams.w > 0.0f);
        if ((!aoEnabled && !contactEnabled) || (width == 0) || (height == 0) || (sceneIndex >= aoSets.size())) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(aoTextures[0].get(), RenderTextureLayout::SHADER_READ));
            return;
        }

        const uint32_t dispatchX = (width + 7) / 8;
        const uint32_t dispatchY = (height + 7) / 8;
        const bool multisampling = (scene.depthTarget->multisampling.sampleCount > 1);
        interop::LightingAOCB aoCB;
        aoCB.sceneIndex = sceneIndex;
        aoCB.outputSize = { width, height };
        aoCB.frameIndex = frameIndex;
        // The depth and the normal buffer were last made readable for the graphics passes.
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, {
            RenderTextureBarrier(scene.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ),
            RenderTextureBarrier(normalBuffer.get(), RenderTextureLayout::SHADER_READ)
        });

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(aoTextures[0].get(), RenderTextureLayout::GENERAL));
        worker->commandList->setPipeline(multisampling ? aoPipelineMS.get() : aoPipeline.get());
        worker->commandList->setComputePipelineLayout(aoPipelineLayout.get());
        worker->commandList->setComputePushConstants(0, &aoCB);
        worker->commandList->setComputeDescriptorSet(aoSets[sceneIndex]->get(), 0);
        worker->commandList->dispatch(dispatchX, dispatchY, 1);

        // Horizontal and vertical blur, ending back in the first texture.
        interop::LightingAOBlurCB blurCB;
        blurCB.size = { width, height };
        worker->commandList->setPipeline(aoBlurPipeline.get());
        worker->commandList->setComputePipelineLayout(aoBlurPipelineLayout.get());
        for (uint32_t pass = 0; pass < 2; pass++) {
            RenderTexture *input = aoTextures[pass].get();
            RenderTexture *output = aoTextures[pass ^ 1].get();
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, { RenderTextureBarrier(input, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(output, RenderTextureLayout::GENERAL) });
            blurCB.direction = (pass == 0) ? interop::int2(1, 0) : interop::int2(0, 1);
            worker->commandList->setComputePushConstants(0, &blurCB);
            worker->commandList->setComputeDescriptorSet(aoBlurSets[pass]->get(), 0);
            worker->commandList->dispatch(dispatchX, dispatchY, 1);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(aoTextures[0].get(), RenderTextureLayout::SHADER_READ));
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

    void LightingRenderer::recordEmissive(RenderWorker *worker, uint32_t sceneIndex) {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        const uint32_t sceneWidth = uint32_t(std::max(scene.rect.right - scene.rect.left, 0));
        const uint32_t sceneHeight = uint32_t(std::max(scene.rect.bottom - scene.rect.top, 0));
        const uint32_t width = std::min((sceneWidth + 3) / 4, emissiveTextureWidth);
        const uint32_t height = std::min((sceneHeight + 3) / 4, emissiveTextureHeight);
        const RenderTexture *sceneColor = nullptr;
        if ((scene.params.emissiveParams.w > 0.0f) && (width > 0) && (height > 0) && (sceneIndex < emissiveSets.size())) {
            sceneColor = copyColor(worker, sceneIndex);
            gpuMarker(worker->commandList.get(), "emissive copy");
        }

        // The composition reads both textures even when the scene doesn't glow.
        if (sceneColor == nullptr) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                RenderTextureBarrier(colorCopyTexture.get(), RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(emissiveTextures[0].get(), RenderTextureLayout::SHADER_READ)
            });

            return;
        }

        const uint32_t dispatchX = (width + 7) / 8;
        const uint32_t dispatchY = (height + 7) / 8;
        const bool multisampling = (scene.depthTarget->multisampling.sampleCount > 1);
        interop::LightingEmissiveCB emissiveCB;
        emissiveCB.sceneIndex = sceneIndex;
        emissiveCB.outputSize = { width, height };
        emissiveCB.padding = 0;
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, {
            RenderTextureBarrier(scene.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ),
            RenderTextureBarrier(colorCopyTexture.get(), RenderTextureLayout::SHADER_READ)
        });

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(emissiveTextures[0].get(), RenderTextureLayout::GENERAL));
        worker->commandList->setPipeline(multisampling ? emissivePipelineMS.get() : emissivePipeline.get());
        worker->commandList->setComputePipelineLayout(emissivePipelineLayout.get());
        worker->commandList->setComputePushConstants(0, &emissiveCB);
        worker->commandList->setComputeDescriptorSet(emissiveSets[sceneIndex]->get(), 0);
        worker->commandList->dispatch(dispatchX, dispatchY, 1);

        // Horizontal and vertical blur, ending back in the first texture.
        interop::LightingEmissiveBlurCB blurCB;
        blurCB.size = { width, height };
        blurCB.radius = uint32_t(std::clamp(enhancementValue("RT64_LIGHT_EMISSIVE_RADIUS", 24.0f), 1.0f, 64.0f));
        blurCB.padding = { 0, 0, 0 };
        worker->commandList->setPipeline(emissiveBlurPipeline.get());
        worker->commandList->setComputePipelineLayout(emissiveBlurPipelineLayout.get());
        for (uint32_t pass = 0; pass < 2; pass++) {
            RenderTexture *input = emissiveTextures[pass].get();
            RenderTexture *output = emissiveTextures[pass ^ 1].get();
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, { RenderTextureBarrier(input, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(output, RenderTextureLayout::GENERAL) });
            blurCB.direction = (pass == 0) ? interop::int2(1, 0) : interop::int2(0, 1);
            worker->commandList->setComputePushConstants(0, &blurCB);
            worker->commandList->setComputeDescriptorSet(emissiveBlurSets[pass]->get(), 0);
            worker->commandList->dispatch(dispatchX, dispatchY, 1);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(emissiveTextures[0].get(), RenderTextureLayout::SHADER_READ));
        gpuMarker(worker->commandList.get(), "emissive");
    }

    void LightingRenderer::recordCompose(RenderWorker *worker, uint32_t sceneIndex) {
        assert(sceneIndex < scenes.size());
        const Scene &scene = scenes[sceneIndex];
        if (scene.rect.isEmpty() || (sceneIndex >= composeSets.size())) {
            return;
        }

        recordEmissive(worker, sceneIndex);

        RenderTarget *colorTarget = scene.colorTarget;
        ComposePipelines &pipelines = getComposePipelines(colorTarget->multisampling, colorTarget->format);
        colorTarget->setupColorFramebuffer(worker);

        // The low quality preset lights the nearest surface of each pixel for all its samples in a single pass.
        const bool multisampling = (colorTarget->multisampling.sampleCount > 1);
        const bool singlePass = multisampling && (getRasterLightingQuality() == 0);
        interop::LightingComposeCB composeCB;
        composeCB.sceneIndex = sceneIndex;
        composeCB.surfacePass = singlePass ? 2 : 0;
        composeCB.padding = { 0, 0 };
        worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(colorTarget->width), float(colorTarget->height)));
        worker->commandList->setScissors(scene.rect);
        worker->commandList->setPipeline((scene.params.settings.x != 0) ? pipelines.copy.get() : pipelines.multiply.get());
        worker->commandList->setGraphicsPipelineLayout(composePipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(composeSets[sceneIndex]->get(), 0);
        worker->commandList->setGraphicsPushConstants(0, &composeCB);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // The farther surfaces of the edge pixels.
        if (multisampling && !singlePass) {
            composeCB.surfacePass = 1;
            worker->commandList->setGraphicsPushConstants(0, &composeCB);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }
    }

    bool LightingRenderer::empty() const {
        return scenes.empty();
    }
};
