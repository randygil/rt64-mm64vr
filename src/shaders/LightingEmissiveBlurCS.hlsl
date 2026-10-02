//
// RT64
//
// Separable gaussian blur of the light of the glowing surfaces (quarter resolution), wide enough to spread it over the
// walls and floors around them. Taps outside the texture count as darkness, so lights at the edges of the screen don't
// get stronger.
//

#include "shared/rt64_lighting_params.h"

[[vk::push_constant]] ConstantBuffer<LightingEmissiveBlurCB> gConstants : register(b0, space0);
Texture2D<float4> gInput : register(t1, space0);
[[vk::image_format("rgba16f")]] RWTexture2D<float4> gOutput : register(u2, space0);

[numthreads(8, 8, 1)]
void CSMain(uint2 threadId : SV_DispatchThreadID) {
    if (any(threadId >= gConstants.size)) {
        return;
    }

    const int radius = int(gConstants.radius);
    const float sigma = max(float(radius) / 2.5f, 0.5f);
    const float exponent = -0.5f / (sigma * sigma);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        const float weight = exp(float(i * i) * exponent);
        weightSum += weight;
        const int2 samplePixel = int2(threadId) + gConstants.direction * i;
        if (all(samplePixel >= 0) && all(samplePixel < int2(gConstants.size))) {
            sum += gInput.Load(int3(samplePixel, 0)).rgb * weight;
        }
    }

    gOutput[threadId] = float4(sum / weightSum, 1.0f);
}
