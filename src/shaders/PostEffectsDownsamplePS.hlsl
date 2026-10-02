//
// RT64
//
// Downsamples a level of the bloom chain to the next one, at half its resolution.
//

#include "PostEffectsCommon.hlsli"
#include "PostEffectsParams.hlsli"

[[vk::push_constant]] ConstantBuffer<PostEffectsDownsampleCB> gConstants : register(b0, space0);
Texture2D<float4> gInput : register(t1, space0);
SamplerState gLinearSampler : register(s2, space0);

float3 sampleInput(float2 position) {
    return gInput.SampleLevel(gLinearSampler, postEffectsClampedUV(position, gConstants.inputSize, gConstants.inputInvSize), 0).rgb;
}

float4 PSMain(in float4 pixelPosition : SV_POSITION) : SV_TARGET {
    const float2 center = pixelPosition.xy * 2.0f;
    if (gConstants.flags & POST_EFFECTS_FLAG_WIDE_FILTER) {
        float3 s[13];
        [unroll]
        for (uint i = 0; i < 13; i++) {
            s[i] = sampleInput(center + PostEffectsWideOffsets[i]);
        }

        return float4(postEffectsCombineWide(s, false), 1.0f);
    }
    else {
        float3 s[5];
        [unroll]
        for (uint i = 0; i < 5; i++) {
            s[i] = sampleInput(center + PostEffectsNarrowOffsets[i]);
        }

        return float4(postEffectsCombineNarrow(s), 1.0f);
    }
}
