//
// RT64
//

#include "rt64_lighting_sky.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>

#include "shaders/FullScreenVS.hlsl.spirv.h"
#include "shaders/LightingSkyLutPS.hlsl.spirv.h"
#include "shaders/LightingSkyPS.hlsl.spirv.h"
#include "shaders/LightingSkyPSMS.hlsl.spirv.h"
#include "shaders/LightingSkyAnalyzeCS.hlsl.spirv.h"
#include "shaders/LightingSkyAnalyzeCSMS.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/FullScreenVS.hlsl.dxil.h"
#   include "shaders/LightingSkyLutPS.hlsl.dxil.h"
#   include "shaders/LightingSkyPS.hlsl.dxil.h"
#   include "shaders/LightingSkyPSMS.hlsl.dxil.h"
#   include "shaders/LightingSkyAnalyzeCS.hlsl.dxil.h"
#   include "shaders/LightingSkyAnalyzeCSMS.hlsl.dxil.h"
#elif defined(__APPLE__)
#   include "shaders/FullScreenVS.hlsl.metal.h"
#   include "shaders/LightingSkyLutPS.hlsl.metal.h"
#   include "shaders/LightingSkyPS.hlsl.metal.h"
#   include "shaders/LightingSkyPSMS.hlsl.metal.h"
#   include "shaders/LightingSkyAnalyzeCS.hlsl.metal.h"
#   include "shaders/LightingSkyAnalyzeCSMS.hlsl.metal.h"
#endif

#include "shaders/LightingSkyCommon.hlsli"

#include "rt64_lighting.h"
#include "rt64_tuning.h"

#ifdef _WIN32
#   define LIGHTING_SKY_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? NAME##BlobDXIL : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? sizeof(NAME##BlobDXIL) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#elif defined(__APPLE__)
#   define LIGHTING_SKY_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? NAME##BlobMSL : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? sizeof(NAME##BlobMSL) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#else
#   define LIGHTING_SKY_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#endif

namespace RT64 {
    // Parameters are written to a ring of slots of an upload buffer. The worker waits for the GPU at the end of every
    // frame, so a slot can be reused as long as a frame doesn't draw the sky more times than there are slots.
    static const uint32_t SkySlotCount = 32;

    // Reading the settings takes a lock and looks up the environment, so they're only read again every few skies.
    static const uint32_t SkySettingsInterval = 8;

    // Altitude of the observer in the atmosphere, in meters.
    static const float SkyObserverAltitude = 200.0f;

    // Exposure that brings the radiance of the atmosphere to the brightness of the skies of the games.
    static const float SkyBaseExposure = 10.8f;

    // Speed of the clouds in cells of their noise per second, before the speed setting.
    static const double SkyBaseCloudSpeed = 0.015;

    // Part of the speed of the clouds the noise that warps their shapes moves at, in its own cells.
    static const double SkyWarpSpeed = 0.1;

    // The noise of the clouds repeats every 256 cells, and the warp is sampled at a quarter of their frequency.
    static const double SkyNoisePeriod = 256.0;
    static const double SkyOriginPeriod = 1024.0;

    // Light bounced by the ground onto the clouds, relative to the ground light of the lighting.
    static const float SkyGroundBounce = 0.65f;

    // Scenes of a frame whose game sky is analyzed separately (see LightingSkyAnalyzeCS).
    static const uint32_t SkyAnalysisCount = 8;

    struct LightingSkyDescriptorSet : RenderDescriptorSetBase {
        uint32_t gSkyParams;
        uint32_t gDepth;
        uint32_t gSceneColor;
        uint32_t gSkyLut;
        uint32_t gLinearSampler;
        uint32_t gSkyAnalysis;

        LightingSkyDescriptorSet(const RenderSampler *linearSampler, RenderDevice *device = nullptr) {
            builder.begin();
            gSkyParams = builder.addStructuredBuffer(1);
            gDepth = builder.addTexture(2);
            gSceneColor = builder.addTexture(3);
            gSkyLut = builder.addTexture(4);
            gLinearSampler = builder.addImmutableSampler(5, linearSampler);
            gSkyAnalysis = builder.addStructuredBuffer(6);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct LightingSkyAnalysisDescriptorSet : RenderDescriptorSetBase {
        uint32_t gSkyParams;
        uint32_t gDepth;
        uint32_t gSceneColor;
        uint32_t gAnalysis;

        LightingSkyAnalysisDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gSkyParams = builder.addStructuredBuffer(1);
            gDepth = builder.addTexture(2);
            gSceneColor = builder.addTexture(3);
            gAnalysis = builder.addReadWriteStructuredBuffer(4);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // Values that can be tuned live (see docs/rendering/procedural-sky.md).
    struct SkySettings {
        int quality;
        uint32_t debugView;
        bool replaceAll;
        float exposure;
        float saturation;
        float tint[3];
        float range;
        float haze;
        float multipleScattering;
        float ozone;
        float sunSize;
        float sunDisc;
        float sunGlow;
        float sunTint;
        float sunReddening;
        float cloudCoverage;
        float cloudOpacity;
        float cloudScale;
        float cloudHeight;
        float cloudCurvature;
        float cloudSpeed;
        float windAngle;
        float cloudWarp;
        float cloudShadow;
        float cloudLight;
        float cloudAmbient;
        float cloudBelly;
        float cloudHaze;
        float groundShadow;
        float groundShadowScale;
        float groundShadowSoftness;
        float horizonBlend;
        float horizonHeight;
        float keyWhite;
        float keyMinLuma;
        float knee;
        float dither;
    };

    static SkySettings readSkySettings() {
        SkySettings s;
        s.quality = int(enhancementValue("RT64_SKY_QUALITY", -1.0f));
        s.debugView = uint32_t(std::max(enhancementValue("RT64_SKY_DEBUG", 0.0f), 0.0f));
        s.replaceAll = enhancementValue("RT64_SKY_REPLACE_ALL", 0.0f) > 0.0f;
        s.exposure = std::max(enhancementValue("RT64_SKY_EXPOSURE", 1.0f), 0.0f);
        s.saturation = std::max(enhancementValue("RT64_SKY_SATURATION", 2.0f), 0.0f);
        s.tint[0] = std::max(enhancementValue("RT64_SKY_TINT_R", 0.85f), 0.0f);
        s.tint[1] = std::max(enhancementValue("RT64_SKY_TINT_G", 0.97f), 0.0f);
        s.tint[2] = std::max(enhancementValue("RT64_SKY_TINT_B", 1.12f), 0.0f);
        s.range = std::clamp(enhancementValue("RT64_SKY_RANGE", 0.75f), 0.1f, 2.0f);
        s.haze = std::max(enhancementValue("RT64_SKY_HAZE", 0.8f), 0.0f);
        s.multipleScattering = std::max(enhancementValue("RT64_SKY_MULTI_SCATTERING", 0.3f), 0.0f);
        s.ozone = std::max(enhancementValue("RT64_SKY_OZONE", 1.0f), 0.0f);
        s.sunSize = std::max(enhancementValue("RT64_SKY_SUN_SIZE", 0.012f), 1e-4f);
        s.sunDisc = std::max(enhancementValue("RT64_SKY_SUN_DISC", 6.0f), 0.0f);
        s.sunGlow = std::max(enhancementValue("RT64_SKY_SUN_GLOW", 1.0f), 0.0f);
        s.sunTint = std::clamp(enhancementValue("RT64_SKY_SUN_TINT", 0.5f), 0.0f, 1.0f);
        s.sunReddening = std::max(enhancementValue("RT64_SKY_SUN_REDDENING", 0.6f), 0.0f);
        s.cloudCoverage = std::clamp(enhancementValue("RT64_SKY_CLOUD_COVERAGE", 0.45f), 0.0f, 1.5f);
        s.cloudOpacity = std::max(enhancementValue("RT64_SKY_CLOUD_OPACITY", 1.0f), 0.0f);
        s.cloudScale = std::max(enhancementValue("RT64_SKY_CLOUD_SCALE", 1.0f), 0.01f);
        s.cloudHeight = std::max(enhancementValue("RT64_SKY_CLOUD_HEIGHT", 30000.0f), 1.0f);
        s.cloudCurvature = std::max(enhancementValue("RT64_SKY_CLOUD_CURVATURE", 12.0f), 0.5f);
        s.cloudSpeed = enhancementValue("RT64_SKY_CLOUD_SPEED", 1.0f);
        s.windAngle = enhancementValue("RT64_SKY_WIND_ANGLE", 20.0f);
        s.cloudWarp = std::max(enhancementValue("RT64_SKY_CLOUD_WARP", 0.3f), 0.0f);
        s.cloudShadow = std::max(enhancementValue("RT64_SKY_CLOUD_SHADOW", 1.0f), 0.0f);
        s.cloudLight = std::max(enhancementValue("RT64_SKY_CLOUD_LIGHT", 1.0f), 0.0f);
        s.cloudAmbient = std::max(enhancementValue("RT64_SKY_CLOUD_AMBIENT", 0.55f), 0.0f);
        s.cloudBelly = std::clamp(enhancementValue("RT64_SKY_CLOUD_BELLY", 0.5f), 0.0f, 1.0f);
        s.cloudHaze = std::max(enhancementValue("RT64_SKY_CLOUD_HAZE", 0.15f), 0.0f);
        s.groundShadow = std::clamp(enhancementValue("RT64_SKY_GROUND_SHADOW", 0.45f), 0.0f, 1.0f);
        s.groundShadowScale = std::max(enhancementValue("RT64_SKY_GROUND_SHADOW_SCALE", 12.0f), 0.01f);
        s.groundShadowSoftness = std::clamp(enhancementValue("RT64_SKY_GROUND_SHADOW_SOFTNESS", 0.06f), 0.01f, 0.5f);
        s.horizonBlend = std::clamp(enhancementValue("RT64_SKY_HORIZON_BLEND", 0.8f), 0.0f, 1.0f);
        s.horizonHeight = std::max(enhancementValue("RT64_SKY_HORIZON_HEIGHT", 0.12f), 1e-3f);
        s.keyWhite = std::max(enhancementValue("RT64_SKY_KEY_WHITE", 1.0f), 0.0f);
        s.keyMinLuma = std::clamp(enhancementValue("RT64_SKY_KEY_MIN_LUMA", 0.18f), 0.001f, 1.0f);
        s.knee = std::clamp(enhancementValue("RT64_SKY_KNEE", 0.6f), 0.0f, 0.95f);
        s.dither = std::max(enhancementValue("RT64_SKY_DITHER", 1.0f), 0.0f);
        return s;
    }

    struct SkyVector {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;

        SkyVector() = default;
        SkyVector(double x, double y, double z) : x(x), y(y), z(z) { }

        SkyVector operator+(const SkyVector &v) const {
            return SkyVector(x + v.x, y + v.y, z + v.z);
        }

        SkyVector operator-(const SkyVector &v) const {
            return SkyVector(x - v.x, y - v.y, z - v.z);
        }

        SkyVector operator*(double s) const {
            return SkyVector(x * s, y * s, z * s);
        }

        double dot(const SkyVector &v) const {
            return x * v.x + y * v.y + z * v.z;
        }

        SkyVector cross(const SkyVector &v) const {
            return SkyVector(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
        }

        double length() const {
            return sqrt(dot(*this));
        }

        interop::float4 toFloat4(float w) const {
            return interop::float4(float(x), float(y), float(z), w);
        }
    };

    static SkyVector skyVector(const interop::float4 &v) {
        return SkyVector(v.x, v.y, v.z);
    }

    static double angleBetween(const SkyVector &a, const SkyVector &b) {
        return atan2(a.cross(b).length(), a.dot(b));
    }

    static double wrapPeriod(double value, double period) {
        return value - floor(value / period) * period;
    }

    // Row vector times matrix, which is what mul(matrix, vector) does with the matrices of the lighting parameters.
    static void transformRow(const interop::float4x4 &m, const double v[4], double result[4]) {
        for (int j = 0; j < 4; j++) {
            result[j] = v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j] + v[3] * m[3][j];
        }
    }

    // Direction of the view ray of a pixel as an affine function of its position (origin + x * stepX + y * stepY) in the
    // space of the geometry, scaled so the ray of the given pixel has unit length. The world points of a pixel are the
    // combinations of a point A at any depth and the camera C (both homogeneous), so A.xyz * C.w - C.xyz * A.w points
    // along the ray, and A is an affine function of the pixel. Returns false if the projection isn't a perspective.
    static bool computeViewRays(const interop::LightingParams &lighting, double centerX, double centerY, SkyVector &origin, SkyVector &stepX, SkyVector &stepY) {
        const interop::float4x4 &invViewProj = lighting.invViewProj;
        const double cameraClip[4] = { 0.0, 0.0, 1.0, 0.0 };
        double camera[4];
        transformRow(invViewProj, cameraClip, camera);

        const double cameraScale = sqrt(camera[0] * camera[0] + camera[1] * camera[1] + camera[2] * camera[2] + camera[3] * camera[3]);
        if (!(fabs(camera[3]) > (cameraScale * 1e-9))) {
            return false;
        }

        auto rayOf = [&](const double point[4]) {
            return SkyVector(point[0] * camera[3] - camera[0] * point[3], point[1] * camera[3] - camera[1] * point[3], point[2] * camera[3] - camera[2] * point[3]);
        };

        const interop::float4 &pixelToClip = lighting.pixelToClip;
        const double originClip[4] = { pixelToClip.z, pixelToClip.w, 0.0, 1.0 };
        const double stepXClip[4] = { pixelToClip.x, 0.0, 0.0, 0.0 };
        const double stepYClip[4] = { 0.0, pixelToClip.y, 0.0, 0.0 };
        double originPoint[4], stepXPoint[4], stepYPoint[4];
        transformRow(invViewProj, originClip, originPoint);
        transformRow(invViewProj, stepXClip, stepXPoint);
        transformRow(invViewProj, stepYClip, stepYPoint);
        origin = rayOf(originPoint);
        stepX = rayOf(stepXPoint);
        stepY = rayOf(stepYPoint);

        // The rays must point away from the camera, towards a point in front of it.
        const double centerClip[4] = { centerX * pixelToClip.x + pixelToClip.z, centerY * pixelToClip.y + pixelToClip.w, 0.5, 1.0 };
        double centerPoint[4];
        transformRow(invViewProj, centerClip, centerPoint);
        if (!(fabs(centerPoint[3]) > 1e-12)) {
            return false;
        }

        const SkyVector cameraPosition(camera[0] / camera[3], camera[1] / camera[3], camera[2] / camera[3]);
        const SkyVector forward = SkyVector(centerPoint[0] / centerPoint[3], centerPoint[1] / centerPoint[3], centerPoint[2] / centerPoint[3]) - cameraPosition;
        SkyVector center = origin + stepX * centerX + stepY * centerY;
        if (center.dot(forward) < 0.0) {
            origin = origin * -1.0;
            stepX = stepX * -1.0;
            stepY = stepY * -1.0;
            center = center * -1.0;
        }

        const double centerLength = center.length();
        if (!(centerLength > 0.0) || !std::isfinite(centerLength)) {
            return false;
        }

        origin = origin * (1.0 / centerLength);
        stepX = stepX * (1.0 / centerLength);
        stepY = stepY * (1.0 / centerLength);
        return true;
    }

    // Fraction of the sunlight that reaches the observer, with the same atmosphere the sky-view table uses.
    static SkyVector computeSunTransmittance(double sunSine, double haze, double ozone) {
        const double PlanetRadius = 6360e3;
        const double AtmosphereRadius = 6460e3;
        const double RayleighScattering[3] = { 5.802e-6, 13.558e-6, 33.1e-6 };
        const double MieExtinction = 4.44e-6;
        const double OzoneAbsorption[3] = { 0.650e-6, 1.881e-6, 0.085e-6 };
        const int Steps = 16;
        const double altitude = SkyObserverAltitude;
        const double radius = PlanetRadius + altitude;
        const double sunCosine = sqrt(std::max(1.0 - sunSine * sunSine, 0.0));
        const double b = radius * sunSine;

        // The planet hides the sun once it's below the horizon.
        const double planetC = altitude * (2.0 * PlanetRadius + altitude);
        if ((b < 0.0) && ((b * b - planetC) >= 0.0)) {
            return SkyVector(0.0, 0.0, 0.0);
        }

        const double atmosphereC = (radius - AtmosphereRadius) * (radius + AtmosphereRadius);
        const double root = sqrt(std::max(b * b - atmosphereC, 0.0));
        const double pathLength = (b > 0.0) ? (-atmosphereC / (b + root)) : (root - b);
        const double stepSize = pathLength / Steps;
        double opticalDepth[3] = { 0.0, 0.0, 0.0 };
        for (int i = 0; i < Steps; i++) {
            const double t = stepSize * (i + 0.5);
            const double x = sunCosine * t;
            const double y = radius + sunSine * t;
            const double sampleAltitude = std::max(sqrt(x * x + y * y) - PlanetRadius, 0.0);
            const double rayleigh = exp(-sampleAltitude / 8e3);
            const double mie = exp(-sampleAltitude / 1.2e3) * haze;
            const double ozoneDensity = std::max(1.0 - fabs(sampleAltitude - 25e3) / 15e3, 0.0) * ozone;
            for (int c = 0; c < 3; c++) {
                opticalDepth[c] += (RayleighScattering[c] * rayleigh + MieExtinction * mie + OzoneAbsorption[c] * ozoneDensity) * stepSize;
            }
        }

        return SkyVector(exp(-opticalDepth[0]), exp(-opticalDepth[1]), exp(-opticalDepth[2]));
    }

    static bool lutConstantsChanged(const interop::LightingSkyLutCB &a, const interop::LightingSkyLutCB &b) {
        // The direction of the sun changes slightly with the rotation of the camera due to rounding.
        const float SunTolerance = 1e-4f;
        if ((fabsf(a.sunDirection.x - b.sunDirection.x) > SunTolerance) || (fabsf(a.sunDirection.y - b.sunDirection.y) > SunTolerance)) {
            return true;
        }

        return (a.sunDirection.w != b.sunDirection.w) || (memcmp(&a.skyTint, &b.skyTint, sizeof(a.skyTint)) != 0) ||
            (memcmp(&a.atmosphere, &b.atmosphere, sizeof(a.atmosphere)) != 0) || (memcmp(&a.observer, &b.observer, sizeof(a.observer)) != 0);
    }

    // LightingSky::Impl

    struct LightingSky::Impl {
        RenderDevice *device = nullptr;
        std::unique_ptr<RenderSampler> linearSampler;
        std::unique_ptr<RenderShader> fullScreenVertexShader;
        std::unique_ptr<RenderShader> skyPixelShader;
        std::unique_ptr<RenderShader> skyPixelShaderMS;
        std::unique_ptr<RenderShader> lutPixelShader;
        std::unique_ptr<RenderPipelineLayout> skyPipelineLayout;
        std::unique_ptr<RenderPipelineLayout> lutPipelineLayout;
        std::unique_ptr<RenderPipeline> lutPipeline;
        std::map<std::pair<uint32_t, RenderFormat>, std::unique_ptr<RenderPipeline>> skyPipelines;
        std::unique_ptr<RenderTexture> lutTexture;
        std::unique_ptr<RenderFramebuffer> lutFramebuffer;
        std::unique_ptr<RenderBuffer> paramsBuffer;
        std::vector<std::unique_ptr<LightingSkyDescriptorSet>> descriptorSets;
        std::unique_ptr<RenderPipelineLayout> analysisPipelineLayout;
        std::unique_ptr<RenderPipeline> analysisPipeline;
        std::unique_ptr<RenderPipeline> analysisPipelineMS;
        std::unique_ptr<RenderBuffer> analysisBuffer;
        std::vector<std::unique_ptr<LightingSkyAnalysisDescriptorSet>> analysisSets;
        bool analysisValid[SkyAnalysisCount] = {};

        // The analysis buffer starts with undefined contents: it's cleared by the first sky drawn.
        bool analysisCleared = false;
        interop::LightingSkyLutCB lutConstants = {};
        bool lutRendered = false;
        uint32_t nextSlot = 0;
        SkySettings settings = {};
        uint32_t settingsCounter = 0;

        const RenderPipeline *getSkyPipeline(const RenderMultisampling &multisampling, RenderFormat format) {
            std::unique_ptr<RenderPipeline> &pipeline = skyPipelines[{ multisampling.sampleCount, format }];
            if (pipeline == nullptr) {
                RenderGraphicsPipelineDesc pipelineDesc;
                pipelineDesc.pipelineLayout = skyPipelineLayout.get();
                pipelineDesc.vertexShader = fullScreenVertexShader.get();
                pipelineDesc.pixelShader = (multisampling.sampleCount > 1) ? skyPixelShaderMS.get() : skyPixelShader.get();
                pipelineDesc.renderTargetFormat[0] = format;
                pipelineDesc.renderTargetCount = 1;
                pipelineDesc.multisampling = multisampling;
                pipelineDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
                pipelineDesc.cullMode = RenderCullMode::NONE;

                // The key is the opacity of the sky, and the alpha channel holds the coverage of the RDP, which must be kept.
                RenderBlendDesc &blendDesc = pipelineDesc.renderTargetBlend[0];
                blendDesc.blendEnabled = true;
                blendDesc.srcBlend = RenderBlend::SRC_ALPHA;
                blendDesc.dstBlend = RenderBlend::INV_SRC_ALPHA;
                blendDesc.blendOp = RenderBlendOperation::ADD;
                blendDesc.srcBlendAlpha = RenderBlend::ZERO;
                blendDesc.dstBlendAlpha = RenderBlend::ONE;
                blendDesc.blendOpAlpha = RenderBlendOperation::ADD;
                blendDesc.renderTargetWriteMask = uint8_t(RenderColorWriteEnable::RED) | uint8_t(RenderColorWriteEnable::GREEN) | uint8_t(RenderColorWriteEnable::BLUE);
                pipeline = device->createGraphicsPipeline(pipelineDesc);
            }

            return pipeline.get();
        }

        void recordLut(RenderWorker *worker, const interop::LightingSkyLutCB &constants) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(lutTexture.get(), RenderTextureLayout::COLOR_WRITE));
            worker->commandList->setFramebuffer(lutFramebuffer.get());
            worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(LIGHTING_SKY_LUT_WIDTH), float(LIGHTING_SKY_LUT_HEIGHT)));
            worker->commandList->setScissors(RenderRect(0, 0, LIGHTING_SKY_LUT_WIDTH, LIGHTING_SKY_LUT_HEIGHT));
            worker->commandList->setPipeline(lutPipeline.get());
            worker->commandList->setGraphicsPipelineLayout(lutPipelineLayout.get());
            worker->commandList->setGraphicsPushConstants(0, &constants);
            worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
            worker->commandList->drawInstanced(3, 1, 0, 0);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(lutTexture.get(), RenderTextureLayout::SHADER_READ));
            lutConstants = constants;
            lutRendered = true;
        }
    };

    // LightingSky

    LightingSky::LightingSky(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat) {
        assert(device != nullptr);

        impl = std::make_unique<Impl>();
        impl->device = device;

        RenderSamplerDesc samplerDesc;
        samplerDesc.minFilter = RenderFilter::LINEAR;
        samplerDesc.magFilter = RenderFilter::LINEAR;
        samplerDesc.mipmapMode = RenderMipmapMode::NEAREST;
        samplerDesc.addressU = RenderTextureAddressMode::CLAMP;
        samplerDesc.addressV = RenderTextureAddressMode::CLAMP;
        samplerDesc.addressW = RenderTextureAddressMode::CLAMP;
        impl->linearSampler = device->createSampler(samplerDesc);

        impl->fullScreenVertexShader = device->createShader(LIGHTING_SKY_SHADER_INPUTS(FullScreenVS, "VSMain", shaderFormat));
        impl->skyPixelShader = device->createShader(LIGHTING_SKY_SHADER_INPUTS(LightingSkyPS, "PSMain", shaderFormat));
        impl->skyPixelShaderMS = device->createShader(LIGHTING_SKY_SHADER_INPUTS(LightingSkyPSMS, "PSMain", shaderFormat));
        impl->lutPixelShader = device->createShader(LIGHTING_SKY_SHADER_INPUTS(LightingSkyLutPS, "PSMain", shaderFormat));

        {
            LightingSkyDescriptorSet descriptorSet(impl->linearSampler.get());
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingSkyCB), RenderShaderStageFlag::PIXEL);
            layoutBuilder.addDescriptorSet(descriptorSet);
            layoutBuilder.end();
            impl->skyPipelineLayout = layoutBuilder.create(device);
        }

        // The sky-view table only needs its constants.
        {
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingSkyLutCB), RenderShaderStageFlag::PIXEL);
            layoutBuilder.end();
            impl->lutPipelineLayout = layoutBuilder.create(device);

            RenderGraphicsPipelineDesc pipelineDesc;
            pipelineDesc.pipelineLayout = impl->lutPipelineLayout.get();
            pipelineDesc.vertexShader = impl->fullScreenVertexShader.get();
            pipelineDesc.pixelShader = impl->lutPixelShader.get();
            pipelineDesc.renderTargetFormat[0] = RenderFormat::R16G16B16A16_FLOAT;
            pipelineDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            pipelineDesc.renderTargetCount = 1;
            pipelineDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
            pipelineDesc.cullMode = RenderCullMode::NONE;
            impl->lutPipeline = device->createGraphicsPipeline(pipelineDesc);

            impl->lutTexture = device->createTexture(RenderTextureDesc::ColorTarget(LIGHTING_SKY_LUT_WIDTH, LIGHTING_SKY_LUT_HEIGHT, RenderFormat::R16G16B16A16_FLOAT));
            impl->lutTexture->setName("Lighting Sky LUT");
            const RenderTexture *lutAttachment = impl->lutTexture.get();
            impl->lutFramebuffer = device->createFramebuffer(RenderFramebufferDesc(&lutAttachment, 1));
        }

        impl->paramsBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::LightingSkyParams) * SkySlotCount, RenderBufferFlag::STORAGE));
        for (uint32_t i = 0; i < SkySlotCount; i++) {
            impl->descriptorSets.emplace_back(std::make_unique<LightingSkyDescriptorSet>(impl->linearSampler.get(), device));
            impl->analysisSets.emplace_back(std::make_unique<LightingSkyAnalysisDescriptorSet>(device));
        }

        // Analysis of the game's sky, which tells whether it can be replaced.
        {
            LightingSkyAnalysisDescriptorSet descriptorSet;
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(interop::LightingSkyAnalysisCB), RenderShaderStageFlag::COMPUTE);
            layoutBuilder.addDescriptorSet(descriptorSet);
            layoutBuilder.end();
            impl->analysisPipelineLayout = layoutBuilder.create(device);

            std::unique_ptr<RenderShader> analysisShader = device->createShader(LIGHTING_SKY_SHADER_INPUTS(LightingSkyAnalyzeCS, "CSMain", shaderFormat));
            std::unique_ptr<RenderShader> analysisShaderMS = device->createShader(LIGHTING_SKY_SHADER_INPUTS(LightingSkyAnalyzeCSMS, "CSMain", shaderFormat));
            impl->analysisPipeline = device->createComputePipeline(RenderComputePipelineDesc(impl->analysisPipelineLayout.get(), analysisShader.get(), 64, 1, 1));
            impl->analysisPipelineMS = device->createComputePipeline(RenderComputePipelineDesc(impl->analysisPipelineLayout.get(), analysisShaderMS.get(), 64, 1, 1));
            impl->analysisBuffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(sizeof(interop::float4) * SkyAnalysisCount, RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS));
            impl->analysisBuffer->setName("Lighting Sky Analysis");
            impl->analysisCleared = false;
        }
    }

    LightingSky::~LightingSky() { }

    const RenderBuffer *LightingSky::getAnalysisBuffer() const {
        return impl->analysisBuffer.get();
    }

    bool LightingSky::isAnalysisValid(uint32_t sceneIndex) const {
        return impl->analysisCleared && impl->analysisValid[sceneIndex % SkyAnalysisCount] && !impl->settings.replaceAll;
    }

    void LightingSky::getCloudShadow(float time, int quality, interop::float4 &params, interop::float4 &offset, interop::float4 &misc) {
        params = interop::float4(0.0f, 0.0f, 0.0f, 0.0f);
        offset = interop::float4(0.0f, 0.0f, 0.0f, 0.0f);
        misc = interop::float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (impl->settingsCounter == 0) {
            impl->settings = readSkySettings();
            impl->settingsCounter = 1;
        }

        // Off on the low quality preset (integrated GPUs, standalone headsets), where every per pixel cost counts.
        const SkySettings &settings = impl->settings;
        if ((quality <= 0) || (settings.groundShadow <= 0.0f) || (settings.cloudOpacity <= 0.0f) || (settings.cloudCoverage <= 0.0f)) {
            return;
        }

        // The same wind as the clouds of the sky (see record). The shadows are smaller than the clouds overhead would
        // cast (by the scale setting) so they pass over the ground at a size that reads in a small game area.
        const double windAngle = double(settings.windAngle) * (3.14159265358979323846 / 180.0);
        const double windX = -cos(windAngle);
        const double windZ = -sin(windAngle);
        const double travel = double(time) * SkyBaseCloudSpeed * double(settings.cloudSpeed);
        const double warpTravel = travel * SkyWarpSpeed;
        const float frequency = (1.0f / settings.cloudScale) * settings.groundShadowScale;
        params = interop::float4(1.0f / settings.cloudHeight, frequency, settings.cloudCoverage, settings.groundShadow * std::min(settings.cloudOpacity, 1.0f));
        offset = interop::float4(
            float(wrapPeriod(windX * travel, SkyNoisePeriod)), float(wrapPeriod(windZ * travel, SkyNoisePeriod)),
            float(wrapPeriod(windX * warpTravel, SkyNoisePeriod)), float(wrapPeriod(windZ * warpTravel, SkyNoisePeriod)));
        misc = interop::float4(settings.cloudWarp, settings.groundShadowSoftness, 3.0f, 0.0f);
    }

    bool LightingSky::enabled() const {
        // The host's sky option is shared with the path tracer's procedural sky.
        return isProceduralSkyEnabled() && (enhancementValue("RT64_SKY_ENABLE", 1.0f) > 0.0f);
    }

    void LightingSky::record(RenderWorker *worker, const LightingSkyDesc &desc) {
        assert(worker != nullptr);

        if ((desc.colorTarget == nullptr) || (desc.depthTarget == nullptr) || (desc.lighting == nullptr) || (desc.sceneColor == nullptr) || desc.rect.isEmpty()) {
            return;
        }

        const interop::LightingParams &lighting = *desc.lighting;
        if (lighting.sunDirection.w <= 0.0f) {
            return;
        }

        const RenderRect &rect = desc.rect;
        const double centerX = 0.5 * double(rect.left + rect.right);
        const double centerY = 0.5 * double(rect.top + rect.bottom);
        SkyVector rayOrigin, rayStepX, rayStepY;
        if (!computeViewRays(lighting, centerX, centerY, rayOrigin, rayStepX, rayStepY)) {
            return;
        }

        if ((impl->settingsCounter++ % SkySettingsInterval) == 0) {
            impl->settings = readSkySettings();
        }

        const SkySettings &settings = impl->settings;

        // Everything the shaders use is in sky space: the basis of the world, with y pointing up.
        const SkyVector worldRight = skyVector(lighting.worldRight);
        const SkyVector worldUp = skyVector(lighting.worldUp);
        const SkyVector worldForward = skyVector(lighting.worldForward);
        auto toSky = [&](const SkyVector &v) {
            return SkyVector(v.dot(worldRight), v.dot(worldUp), v.dot(worldForward));
        };

        rayOrigin = toSky(rayOrigin);
        rayStepX = toSky(rayStepX);
        rayStepY = toSky(rayStepY);
        const SkyVector centerRay = rayOrigin + rayStepX * centerX + rayStepY * centerY;
        const double pixelAngle = std::max(angleBetween(centerRay, centerRay + rayStepX), angleBetween(centerRay, centerRay + rayStepY));
        SkyVector sun = toSky(skyVector(lighting.sunDirection));
        const double sunLength = sun.length();
        sun = (sunLength > 1e-8) ? (sun * (1.0 / sunLength)) : SkyVector(0.0, 1.0, 0.0);

        // The sky-view table only depends on the elevation of the sun and the look, so it's rendered again only when
        // they change.
        interop::LightingSkyLutCB lutConstants;
        const float sunSine = float(sun.y);
        lutConstants.sunDirection = interop::float4(sqrtf(std::max(1.0f - sunSine * sunSine, 0.0f)), sunSine, 0.0f, SkyBaseExposure * settings.exposure);
        lutConstants.skyTint = interop::float4(settings.tint[0], settings.tint[1], settings.tint[2], settings.saturation);
        lutConstants.atmosphere = interop::float4(settings.haze, settings.multipleScattering, settings.ozone, settings.range);
        lutConstants.observer = interop::float4(SkyObserverAltitude, 0.0f, 0.0f, 0.0f);
        if (!impl->lutRendered || lutConstantsChanged(impl->lutConstants, lutConstants)) {
            impl->recordLut(worker, lutConstants);
        }

        // The sun takes part of the tint of the scene's sun (which may follow the time of day of the game) and is
        // reddened by the atmosphere, softened so the clouds stay white while the sun is high. It's normalized while
        // it's up so the clouds keep their brightness, and fades out as it sets.
        float sunTint[3] = { lighting.sunColor.x, lighting.sunColor.y, lighting.sunColor.z };
        const float sunTintMax = std::max(std::max(sunTint[0], sunTint[1]), sunTint[2]);
        for (float &channel : sunTint) {
            channel = (sunTintMax > 1e-6f) ? (1.0f + (channel / sunTintMax - 1.0f) * settings.sunTint) : 1.0f;
        }

        const SkyVector transmittance = computeSunTransmittance(sun.y, settings.haze, settings.ozone);
        const double transmittanceMax = std::max(std::max(transmittance.x, transmittance.y), transmittance.z);
        const double transmittanceNormalization = 1.0 / std::max(transmittanceMax, 1e-9);
        const double sunFade = std::min(transmittanceMax / 0.05, 1.0);
        const SkyVector sunLight = SkyVector(
            pow(transmittance.x * transmittanceNormalization, double(settings.sunReddening)),
            pow(transmittance.y * transmittanceNormalization, double(settings.sunReddening)),
            pow(transmittance.z * transmittanceNormalization, double(settings.sunReddening))) * sunFade;

        // The clouds stay in place in the world, and move by as the camera moves if the host tells where the world is.
        SkyVector cameraSky(0.0, 0.0, 0.0);
        if (lighting.worldOrigin.w > 0.0f) {
            cameraSky = toSky(skyVector(lighting.cameraPosition) - skyVector(lighting.worldOrigin));
        }

        const double cloudFrequency = 1.0 / double(settings.cloudScale);
        const double originPeriod = SkyOriginPeriod / cloudFrequency;

        // The wind blows in a fixed direction of the world (an angle from its right axis towards its forward axis). An
        // offset added to the coordinates of the noise moves the clouds the opposite way, hence the signs. The offsets
        // wrap at the period of the noise, which is seamless.
        const double windAngle = double(settings.windAngle) * (3.14159265358979323846 / 180.0);
        const double windX = -cos(windAngle);
        const double windZ = -sin(windAngle);
        const double travel = double(desc.time) * SkyBaseCloudSpeed * double(settings.cloudSpeed);
        const double warpTravel = travel * SkyWarpSpeed;

        RenderTarget *colorTarget = desc.colorTarget;
        const float ditherUnit = (colorTarget->format == RenderFormat::R8G8B8A8_UNORM) ? (1.0f / 255.0f) : 0.0f;
        const int quality = (settings.quality < 0) ? getRasterLightingQuality() : settings.quality;
        interop::LightingSkyParams params;
        memset(&params, 0, sizeof(params));
        params.rayOrigin = rayOrigin.toFloat4(lighting.depthToClip.z);
        params.rayStepX = rayStepX.toFloat4(float(pixelAngle));
        params.rayStepY = rayStepY.toFloat4(0.0f);
        params.rect = interop::float4(float(rect.left), float(rect.top), float(rect.right), float(rect.bottom));
        params.sunDirection = sun.toFloat4(settings.sunSize);
        params.sunColor = interop::float4(float(sunTint[0] * sunLight.x), float(sunTint[1] * sunLight.y), float(sunTint[2] * sunLight.z), settings.sunDisc);
        params.groundColor = interop::float4(lighting.groundColor.x * SkyGroundBounce, lighting.groundColor.y * SkyGroundBounce, lighting.groundColor.z * SkyGroundBounce, settings.sunGlow);
        params.cloudOrigin = interop::float4(float(wrapPeriod(cameraSky.x / settings.cloudHeight, originPeriod)), float(wrapPeriod(cameraSky.z / settings.cloudHeight, originPeriod)), float(cloudFrequency), settings.cloudCurvature);
        params.cloudOffset = interop::float4(
            float(wrapPeriod(windX * travel, SkyNoisePeriod)), float(wrapPeriod(windZ * travel, SkyNoisePeriod)),
            float(wrapPeriod(windX * warpTravel, SkyNoisePeriod)), float(wrapPeriod(windZ * warpTravel, SkyNoisePeriod)));

        params.cloudParams = interop::float4(settings.cloudCoverage, settings.cloudOpacity, settings.cloudWarp, settings.cloudShadow);
        params.cloudLight = interop::float4(settings.cloudLight, settings.cloudAmbient, settings.cloudBelly, settings.cloudHaze);
        params.blendParams = interop::float4(settings.horizonBlend, settings.horizonHeight, settings.keyWhite, settings.knee);
        params.outputParams = interop::float4(ditherUnit * settings.dither, settings.keyMinLuma, 0.0f, 0.0f);
        params.settings.x = uint32_t(std::clamp(quality, 0, 3));
        params.settings.y = settings.debugView;
        params.settings.z = std::max(colorTarget->multisampling.sampleCount, 1U);
        // When the host removed the game's sky (VR), the background has nothing worth keeping.
        const bool replaceAll = settings.replaceAll || ((lighting.settings.w & LIGHTING_SCENE_FLAG_SKY_HIDDEN) != 0);
        params.settings.w = replaceAll ? LIGHTING_SKY_FLAG_REPLACE_ALL : 0U;

        const uint32_t slot = impl->nextSlot;
        impl->nextSlot = (slot + 1) % SkySlotCount;
        uint8_t *paramsData = reinterpret_cast<uint8_t *>(impl->paramsBuffer->map());
        memcpy(paramsData + sizeof(interop::LightingSkyParams) * slot, &params, sizeof(params));
        impl->paramsBuffer->unmap();

        LightingSkyDescriptorSet *descriptorSet = impl->descriptorSets[slot].get();
        descriptorSet->setBuffer(descriptorSet->gSkyParams, impl->paramsBuffer.get(), RenderBufferStructuredView(sizeof(interop::LightingSkyParams)));
        descriptorSet->setTexture(descriptorSet->gDepth, desc.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, desc.depthTarget->textureView.get());
        descriptorSet->setTexture(descriptorSet->gSceneColor, desc.sceneColor, RenderTextureLayout::SHADER_READ);
        descriptorSet->setTexture(descriptorSet->gSkyLut, impl->lutTexture.get(), RenderTextureLayout::SHADER_READ);
        descriptorSet->setBuffer(descriptorSet->gSkyAnalysis, impl->analysisBuffer.get(), RenderBufferStructuredView(sizeof(interop::float4)));

        // The game's sky is analyzed first, unless all of it is replaced anyway.
        const uint32_t sceneIndex = desc.sceneIndex % SkyAnalysisCount;
        if (!replaceAll) {
            LightingSkyAnalysisDescriptorSet *analysisSet = impl->analysisSets[slot].get();
            analysisSet->setBuffer(analysisSet->gSkyParams, impl->paramsBuffer.get(), RenderBufferStructuredView(sizeof(interop::LightingSkyParams)));
            analysisSet->setTexture(analysisSet->gDepth, desc.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, desc.depthTarget->textureView.get());
            analysisSet->setTexture(analysisSet->gSceneColor, desc.sceneColor, RenderTextureLayout::SHADER_READ);
            analysisSet->setBuffer(analysisSet->gAnalysis, impl->analysisBuffer.get(), RenderBufferStructuredView(sizeof(interop::float4)));

            interop::LightingSkyAnalysisCB analysisCB;
            analysisCB.slot = slot;
            analysisCB.padding = 0;
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE,
                { RenderBufferBarrier(impl->analysisBuffer.get(), RenderBufferAccess::WRITE) },
                { RenderTextureBarrier(const_cast<RenderTexture *>(desc.sceneColor), RenderTextureLayout::SHADER_READ), RenderTextureBarrier(desc.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ) });
            worker->commandList->setPipeline((colorTarget->multisampling.sampleCount > 1) ? impl->analysisPipelineMS.get() : impl->analysisPipeline.get());
            worker->commandList->setComputePipelineLayout(impl->analysisPipelineLayout.get());
            worker->commandList->setComputeDescriptorSet(analysisSet->get(), 0);

            // The first time, every entry gets a valid value, as the lighting reads them all.
            if (!impl->analysisCleared) {
                for (uint32_t i = 0; i < SkyAnalysisCount; i++) {
                    analysisCB.sceneIndex = i;
                    analysisCB.reset = 1;
                    worker->commandList->setComputePushConstants(0, &analysisCB);
                    worker->commandList->dispatch(1, 1, 1);
                }

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(impl->analysisBuffer.get(), RenderBufferAccess::WRITE));
                impl->analysisCleared = true;
            }

            analysisCB.sceneIndex = sceneIndex;
            analysisCB.reset = impl->analysisValid[sceneIndex] ? 0U : 1U;
            impl->analysisValid[sceneIndex] = true;
            worker->commandList->setComputePushConstants(0, &analysisCB);
            worker->commandList->dispatch(1, 1, 1);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(impl->analysisBuffer.get(), RenderBufferAccess::READ));

        interop::LightingSkyCB skyCB;
        skyCB.slot = slot;
        skyCB.sceneIndex = sceneIndex;
        skyCB.padding = { 0, 0 };
        colorTarget->setupColorFramebuffer(worker);
        worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(colorTarget->width), float(colorTarget->height)));
        worker->commandList->setScissors(rect);
        worker->commandList->setPipeline(impl->getSkyPipeline(colorTarget->multisampling, colorTarget->format));
        worker->commandList->setGraphicsPipelineLayout(impl->skyPipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(descriptorSet->get(), 0);
        worker->commandList->setGraphicsPushConstants(0, &skyCB);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);
    }
};
