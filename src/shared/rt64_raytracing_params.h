//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    enum class VisualizationMode {
        Final,
        ShadingPosition,
        ShadingNormal,
        ShadingSpecular,
        Diffuse,
        InstanceId,
        DirectLightRaw,
        DirectLightFiltered,
        IndirectLightRaw,
        IndirectLightFiltered,
        Reflection,
        Refraction,
        Transparent,
        Flow,
        ReactiveMask,
        LockMask,
        Depth,
        Count
    };

    struct RaytracingParams {
        float4x4 view;
        float4x4 viewI;
        float4x4 projection;
        float4x4 projectionI;
        float4x4 viewProj;
        float4x4 prevViewProj;
        float4 cameraU;
        float4 cameraV;
        float4 cameraW;
        float4 viewport;
        float4 resolution;
        float4 ambientBaseColor;
        float4 ambientNoGIColor;
        float4 eyeLightDiffuseColor;
        float4 eyeLightSpecularColor;
        float4 sunDirection;
        float4 sunColor;
        float4 worldUp;
        float4 worldRight;
        float4 worldForward;
        float4 skyTint;
        float4 worldOrigin;
        float4 skyZenithColor;
        float4 skyHorizonSunColor;
        float4 skyHorizonAwayColor;
        float2 pixelJitter;
        float fovRadians;
        float nearDist;
        float farDist;
        float giDiffuseStrength;
        float giBackgroundStrength;
        float motionBlurStrength;
        float tonemapExposure;
        float tonemapWhite;
        float tonemapBlack;
        uint diSamples;
        uint giSamples;
        uint diReproject;
        uint giReproject;
        uint binaryLockMask;
        uint maxLights;
        uint motionBlurSamples;
        uint lightsCount;
        uint interleavedRastersCount;
        VisualizationMode visualizationMode;
        uint debugFlags;
        float aoRadius;
        float aoStrength;
        float volumetricStrength;
        float volumetricDistance;
        float sunDiscStrength;
        float bloomStrength;
        float bloomThreshold;
        float sharpenStrength;
        float vignetteStrength;
        float saturation;
        float contrast;
        float blobShadowRemoval;
        float directHistoryLength;
        float volumetricAnisotropy;
        float volumetricMaxPhase;
        float bumpStrength;
        float waterReflection;
        float emissiveStrength;
        float emissiveThreshold;
        float skyMode;
        float skyExposure;
        float skySaturation;
        float cloudCoverage;
        float cloudDensity;
        float cloudScale;
        float skyTime;
        float skyGI;
        float textureSmoothing;
        float spriteVolume;
        float cloudHeight;
        float cloudShadows;
        float foliageWind;
        float foliageDetail;

#   ifdef HLSL_CPU
        RaytracingParams() {
            view = FLOAT4X4_IDENTITY;
            viewI = FLOAT4X4_IDENTITY;
            projection = FLOAT4X4_IDENTITY;
            projectionI = FLOAT4X4_IDENTITY;
            viewProj = FLOAT4X4_IDENTITY;
            prevViewProj = FLOAT4X4_IDENTITY;
            cameraU = { 0.0f, 0.0f, 0.0f, 0.0f };
            cameraV = { 0.0f, 0.0f, 0.0f, 0.0f };
            cameraW = { 0.0f, 0.0f, 0.0f, 0.0f };
            viewport = { 0.0f, 0.0f, 0.0f, 0.0f };
            resolution = { 0.0f, 0.0f, 0.0f, 0.0f };
            ambientBaseColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            ambientNoGIColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            eyeLightDiffuseColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            eyeLightSpecularColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            sunDirection = { 0.0f, 0.0f, 0.0f, 0.0f };
            sunColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            pixelJitter = { 0.0f, 0.0f };
            fovRadians = 0.707f;
            nearDist = 1.0f;
            farDist = 1000.0f;
            giDiffuseStrength = 0.0f;
            giBackgroundStrength = 0.0f;
            motionBlurStrength = 0.0f;
            tonemapExposure = 0.6f;
            tonemapWhite = 1.0f;
            tonemapBlack = 0.0f;
            diSamples = 0;
            giSamples = 0;
            diReproject = 0;
            giReproject = 0;
            binaryLockMask = 0;
            maxLights = 1;
            motionBlurSamples = 32;
            visualizationMode = VisualizationMode::Final;
            debugFlags = 0;
            aoRadius = 0.0f;
            aoStrength = 0.0f;
            volumetricStrength = 0.0f;
            volumetricDistance = 0.0f;
            sunDiscStrength = 0.0f;
            bloomStrength = 0.0f;
            bloomThreshold = 1.0f;
            sharpenStrength = 0.0f;
            vignetteStrength = 0.0f;
            saturation = 1.0f;
            contrast = 1.0f;
            blobShadowRemoval = 0.0f;
            directHistoryLength = 8.0f;
            volumetricAnisotropy = 0.0f;
            volumetricMaxPhase = 1.0f;
            bumpStrength = 0.0f;
            waterReflection = 0.0f;
            emissiveStrength = 0.0f;
            emissiveThreshold = 1.0f;
            skyMode = 0.0f;
            skyExposure = 1.0f;
            skySaturation = 1.0f;
            cloudCoverage = 0.0f;
            cloudDensity = 0.0f;
            cloudScale = 1.0f;
            skyTime = 0.0f;
            skyGI = 0.0f;
            textureSmoothing = 0.0f;
            spriteVolume = 0.0f;
            cloudHeight = 1.0f;
            cloudShadows = 0.0f;
            foliageWind = 0.0f;
            foliageDetail = 0.0f;
        }
#   endif
    };
#ifdef HLSL_CPU
};
#endif