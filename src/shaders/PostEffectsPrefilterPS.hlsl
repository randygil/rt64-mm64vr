//
// RT64
//
// First pass of the post effects. Downsamples the scene to half resolution keeping only its bright parts for the bloom,
// and stores in the alpha channel the mask the light shafts are blurred from: the sky (the pixels nothing was drawn on)
// around the sun, weighted by its brightness.
//

#include "PostEffectsCommon.hlsli"
#include "PostEffectsParams.hlsli"

[[vk::push_constant]] ConstantBuffer<PostEffectsPrefilterCB> gConstants : register(b0, space0);
Texture2D<float4> gSceneColor : register(t1, space0);
#ifdef MULTISAMPLING
Texture2DMS<float> gDepth : register(t2, space0);
#else
Texture2D<float> gDepth : register(t2, space0);
#endif
SamplerState gLinearSampler : register(s3, space0);

float loadDepth(int2 pixel) {
    pixel = clamp(pixel, gConstants.sourceRect.xy, gConstants.sourceRect.zw);
#ifdef MULTISAMPLING
    return gDepth.Load(pixel, 0);
#else
    return gDepth.Load(int3(pixel, 0));
#endif
}

float3 sampleScene(float2 position, float2 invTextureSize) {
    const float2 clamped = clamp(position, float2(gConstants.sourceRect.xy) + 0.5f, float2(gConstants.sourceRect.zw) + 0.5f);
    return gSceneColor.SampleLevel(gLinearSampler, clamped * invTextureSize, 0).rgb;
}

// Part of the color above the threshold on its luminance, with a quadratic transition as wide as the knee around it.
float3 brightPart(float3 color) {
    const float luma = postEffectsLuma(color);
    const float knee = gConstants.bloomKnee;
    float soft = clamp(luma - gConstants.bloomThreshold + knee, 0.0f, 2.0f * knee);
    soft = (soft * soft) / (4.0f * knee + 1e-5f);
    return color * (max(soft, luma - gConstants.bloomThreshold) / max(luma, 1e-5f));
}

float4 PSMain(in float4 pixelPosition : SV_POSITION) : SV_TARGET {
    uint textureWidth, textureHeight;
    gSceneColor.GetDimensions(textureWidth, textureHeight);
    const float2 invTextureSize = 1.0f / float2(textureWidth, textureHeight);

    // The output pixel covers the 2x2 pixels of the scene around this corner.
    const float2 center = float2(gConstants.sourceRect.xy) + pixelPosition.xy * 2.0f;
    float3 centerColor;
    float3 bloom;
    if (gConstants.flags & POST_EFFECTS_FLAG_WIDE_FILTER) {
        float3 s[13];
        [unroll]
        for (uint i = 0; i < 13; i++) {
            s[i] = sampleScene(center + PostEffectsWideOffsets[i], invTextureSize);
        }

        centerColor = s[6];
        [unroll]
        for (uint j = 0; j < 13; j++) {
            s[j] = brightPart(s[j]);
        }

        bloom = postEffectsCombineWide(s, true);
    }
    else {
        float3 s[5];
        [unroll]
        for (uint i = 0; i < 5; i++) {
            s[i] = sampleScene(center + PostEffectsNarrowOffsets[i], invTextureSize);
        }

        centerColor = s[0];
        [unroll]
        for (uint j = 0; j < 5; j++) {
            s[j] = brightPart(s[j]);
        }

        bloom = postEffectsCombineNarrow(s);
    }

    float mask = 0.0f;
    if (gConstants.flags & POST_EFFECTS_FLAG_SHAFT_MASK) {
        const int2 corner = int2(center) - 1;
        float background = 0.0f;
        [unroll]
        for (int y = 0; y < 2; y++) {
            [unroll]
            for (int x = 0; x < 2; x++) {
                background += (loadDepth(corner + int2(x, y)) >= gConstants.backgroundDepth) ? 0.25f : 0.0f;
            }
        }

        // Dark backgrounds (night skies, painted scenery) emit little, and the light fades away from the sun.
        const float brightness = saturate((postEffectsLuma(centerColor) - gConstants.shaftThreshold) / max(1.0f - gConstants.shaftThreshold, 1e-3f));
        const float glow = saturate(1.0f - length(center - gConstants.sunPosition) * gConstants.sunInvRadius);
        mask = background * brightness * glow * glow;
    }

    return float4(bloom, mask);
}
