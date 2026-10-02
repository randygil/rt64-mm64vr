//
// RT64
//

#include "Color.hlsli"
#include "Constants.hlsli"
#include "Math.hlsli"

#include "shared/rt64_raytracing_params.h"

#define ENABLE_EXPOSURE_ADJUSTMENT

ConstantBuffer<RaytracingParams> RtParams : register(b0);
Texture2D<float4> gInput : register(t1);
Texture2D<float4> gFlow : register(t2);
Texture2D<float> gLumaAvg : register(t3);
SamplerState gSampler : register(s4);
Texture2D<float4> gBloom : register(t5);

float4 ColorMotionBlurred(float2 uv) {
    if ((RtParams.motionBlurStrength > 0.0f) && (RtParams.motionBlurSamples > 0)) {
        float2 flow = gFlow.SampleLevel(gSampler, uv, 0).xy / RtParams.resolution.xy;
        float flowLength = length(flow);
        if (flowLength > EPSILON) {
            const float SampleStep = RtParams.motionBlurStrength / RtParams.motionBlurSamples;
            float3 sumColor = float3(0.0f, 0.0f, 0.0f);
            float sumWeight = 0.0f;
            float2 startUV = uv - (flow * RtParams.motionBlurStrength / 2.0f);
            for (uint s = 0; s < RtParams.motionBlurSamples; s++) {
                float2 sampleUV = clamp(startUV + flow * s * SampleStep, float2(0.0f, 0.0f), float2(1.0f, 1.0f));
                float sampleWeight = 1.0f;
                float4 outputColor = gInput.SampleLevel(gSampler, sampleUV, 0);
                sumColor += outputColor.rgb * sampleWeight;
                sumWeight += sampleWeight;
            }

            return float4(sumColor / sumWeight, 1.0f);
        }
    }

    float4 outputColor = gInput.SampleLevel(gSampler, uv, 0);
    return float4(outputColor.rgb, 1.0f);
}

float3 WhiteBlackPoint(float3 bl, float3 wp, float3 color) {
    return (color - bl) / (wp - bl);
}

float3 Sanitize(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0.0f, 0.0f, 0.0f) : max(c, 0.0f);
}

// Contrast adaptive sharpening. Restores the detail softened by the temporal accumulation.
float3 Sharpen(float2 uv, float3 center) {
    uint2 dims;
    gInput.GetDimensions(dims.x, dims.y);
    const float2 texel = 1.0f / float2(dims);
    const float3 n = Sanitize(gInput.SampleLevel(gSampler, uv + float2(0.0f, -texel.y), 0).rgb);
    const float3 s = Sanitize(gInput.SampleLevel(gSampler, uv + float2(0.0f, texel.y), 0).rgb);
    const float3 e = Sanitize(gInput.SampleLevel(gSampler, uv + float2(texel.x, 0.0f), 0).rgb);
    const float3 w = Sanitize(gInput.SampleLevel(gSampler, uv + float2(-texel.x, 0.0f), 0).rgb);
    const float3 minColor = min(center, min(min(n, s), min(e, w)));
    const float3 maxColor = max(center, max(max(n, s), max(e, w)));

    // Less sharpening is applied where the neighborhood already has a lot of contrast to avoid halos.
    const float3 amount = sqrt(saturate(min(minColor, 1.0f - min(maxColor, 1.0f)) / max(maxColor, EPSILON)));
    const float3 weight = -amount * lerp(0.0f, 0.2f, saturate(RtParams.sharpenStrength));
    const float3 result = (center + (n + s + e + w) * weight) / (1.0f + 4.0f * weight);
    return clamp(result, minColor, maxColor);
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    // For the reasons stated in ComposePS, the exposure adjustment is performed in sRGB space and to
    // make the histogram have a better distribution since it doesn't use a logarithmic scale.
    float4 color = max(ColorMotionBlurred(uv), 0.0f);
    color.rgb = Sanitize(color.rgb);
    if (RtParams.sharpenStrength > 0.0f) {
        color.rgb = Sharpen(uv, color.rgb);
    }

    if (RtParams.bloomStrength > 0.0f) {
        color.rgb += Sanitize(gBloom.SampleLevel(gSampler, uv, 0).rgb) * RtParams.bloomStrength;
    }

#ifdef ENABLE_EXPOSURE_ADJUSTMENT
    float avgLuma = gLumaAvg[uint2(0, 0)];
    // The adjustment is limited so the image never strays too far from the original brightness of the game.
    float exposure = clamp(RtParams.tonemapExposure / max(avgLuma, EPSILON), 0.7f, 2.0f);
    color.rgb *= exposure;
    color.rgb = WhiteBlackPoint(RtParams.tonemapBlack, RtParams.tonemapWhite, color.rgb);
#endif

    // Highlights above the shoulder are compressed smoothly instead of being clipped.
    const float Shoulder = 0.85f;
    const float3 over = max(color.rgb - Shoulder, 0.0f);
    color.rgb = min(color.rgb, Shoulder) + over / (1.0f + over / (1.0f - Shoulder));

    // Color grading.
    const float luma = dot(color.rgb, float3(0.2126f, 0.7152f, 0.0722f));
    color.rgb = max(lerp(luma.xxx, color.rgb, RtParams.saturation), 0.0f);
    color.rgb = saturate((color.rgb - 0.5f) * RtParams.contrast + 0.5f);

    // Vignette.
    const float2 centered = uv * 2.0f - 1.0f;
    color.rgb *= 1.0f - RtParams.vignetteStrength * smoothstep(0.4f, 2.0f, dot(centered, centered));
    return float4(saturate(color.rgb), 1.0f);
}
