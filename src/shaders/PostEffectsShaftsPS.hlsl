//
// RT64
//
// Light shafts from the sun as a screen space radial blur (Mitchell, "Volumetric Light Scattering as a Post-Process",
// GPU Gems 3, chapter 13): the mask of the sky around the sun is averaged along the segment from each pixel towards the
// sun, so whatever is drawn in between casts streaks of shadow and the gaps between objects let rays through. The first
// pass covers the whole segment and an optional second pass averages over the length of one step of the first one, which
// removes its banding without needing more samples.
//

#include "PostEffectsCommon.hlsli"
#include "PostEffectsParams.hlsli"

[[vk::push_constant]] ConstantBuffer<PostEffectsShaftsCB> gConstants : register(b0, space0);
Texture2D<float4> gInput : register(t1, space0);
SamplerState gLinearSampler : register(s2, space0);

float sampleMask(float2 position) {
    const float4 value = gInput.SampleLevel(gLinearSampler, postEffectsClampedUV(position, gConstants.inputSize, gConstants.inputInvSize), 0);
    return dot(value, gConstants.channelMask);
}

float4 PSMain(in float4 pixelPosition : SV_POSITION) : SV_TARGET {
    const float2 position = pixelPosition.xy * gConstants.inputScale;
    const float2 step = (gConstants.sunPosition - position) * (gConstants.density / float(gConstants.sampleCount));
    const float offset = (gConstants.flags & POST_EFFECTS_FLAG_JITTER) ? postEffectsNoise(pixelPosition.xy) : 0.5f;
    float weight = 1.0f;
    float sum = 0.0f;
    float weightSum = 0.0f;
    [loop]
    for (uint i = 0; i < gConstants.sampleCount; i++) {
        sum += sampleMask(position + step * (float(i) + offset)) * weight;
        weightSum += weight;
        weight *= gConstants.decay;
    }

    const float shafts = sum / max(weightSum, 1e-5f);
    return float4(shafts, shafts, shafts, 1.0f);
}
