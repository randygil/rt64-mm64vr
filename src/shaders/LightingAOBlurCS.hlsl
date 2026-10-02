//
// RT64
//
// Separable depth aware blur of the ambient occlusion: neighbors at a similar distance from the camera are averaged
// with gaussian weights, so the noise of the per pixel rotation goes away without bleeding across edges. The contact
// shadows get a narrower blur to stay sharp.
//

#include "shared/rt64_lighting_params.h"

[[vk::push_constant]] ConstantBuffer<LightingAOBlurCB> gConstants : register(b0, space0);
Texture2D<float4> gInput : register(t1, space0);
[[vk::image_format("rgba16f")]] RWTexture2D<float4> gOutput : register(u2, space0);

static const float Weights[5] = { 0.2270270f, 0.1945946f, 0.1216216f, 0.0540541f, 0.0162162f };

[numthreads(8, 8, 1)]
void CSMain(uint2 threadId : SV_DispatchThreadID) {
    if (any(threadId >= gConstants.size)) {
        return;
    }

    const float4 center = gInput.Load(int3(threadId, 0));
    float sum = center.x * Weights[0];
    float weightSum = Weights[0];
    float contactSum = center.z * Weights[0];
    float contactWeightSum = Weights[0];
    for (int i = 1; i < 5; i++) {
        for (int side = -1; side <= 1; side += 2) {
            const int2 samplePixel = clamp(int2(threadId) + gConstants.direction * (i * side), int2(0, 0), int2(gConstants.size) - 1);
            const float4 value = gInput.Load(int3(samplePixel, 0));
            const float similarity = saturate(1.0f - abs(value.y - center.y) / (center.y * 0.04f + 1.0f));
            const float weight = Weights[i] * similarity;
            sum += value.x * weight;
            weightSum += weight;
            if (i <= 2) {
                contactSum += value.z * weight;
                contactWeightSum += weight;
            }
        }
    }

    gOutput[threadId] = float4(sum / weightSum, center.y, contactSum / contactWeightSum, 1.0f);
}
