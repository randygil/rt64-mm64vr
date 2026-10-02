//
// RT64
//
// Upsamples a level of the bloom chain over the previous one, which is drawn with alpha blending so the result is
// lerp(level, upsampled, scatter). Going up the whole chain adds every level with weights that sum to one, so the wide
// levels spread the light without making the bloom brighter.
//

#include "PostEffectsCommon.hlsli"
#include "PostEffectsParams.hlsli"

[[vk::push_constant]] ConstantBuffer<PostEffectsUpsampleCB> gConstants : register(b0, space0);
Texture2D<float4> gInput : register(t1, space0);
SamplerState gLinearSampler : register(s2, space0);

float3 sampleInput(float2 position) {
    return gInput.SampleLevel(gLinearSampler, postEffectsClampedUV(position, gConstants.inputSize, gConstants.inputInvSize), 0).rgb;
}

float4 PSMain(in float4 pixelPosition : SV_POSITION) : SV_TARGET {
    // Position in the pixels of the smaller level.
    const float2 center = pixelPosition.xy * 0.5f;
    const float r = gConstants.radius;
    float3 color;
    if (gConstants.flags & POST_EFFECTS_FLAG_WIDE_FILTER) {
        // 3x3 tent filter.
        color = sampleInput(center) * 4.0f;
        color += (sampleInput(center + float2(0.0f, -r)) + sampleInput(center + float2(-r, 0.0f)) + sampleInput(center + float2(r, 0.0f)) + sampleInput(center + float2(0.0f, r))) * 2.0f;
        color += sampleInput(center + float2(-r, -r)) + sampleInput(center + float2(r, -r)) + sampleInput(center + float2(-r, r)) + sampleInput(center + float2(r, r));
        color *= (1.0f / 16.0f);
    }
    else {
        // Four bilinear samples halfway to the diagonals cover about the same footprint.
        const float h = r * 0.5f;
        color = sampleInput(center + float2(-h, -h)) + sampleInput(center + float2(h, -h)) + sampleInput(center + float2(-h, h)) + sampleInput(center + float2(h, h));
        color *= 0.25f;
    }

    return float4(color, gConstants.scatter);
}
