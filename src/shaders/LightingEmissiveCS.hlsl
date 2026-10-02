//
// RT64
//
// Light given off by the glowing surfaces of scenes without a sun (lamps, screens, crystals), like the emissive surfaces
// of the path tracer: the color of those surfaces averaged over blocks of 4x4 pixels into a quarter resolution texture,
// which LightingEmissiveBlurCS spreads over their surroundings before the composition adds it as light.
//

#include "shared/rt64_lighting_params.h"

#include "LightingCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingEmissiveCB> gConstants : register(b0, space0);
StructuredBuffer<LightingParams> gLightingParams : register(t1, space0);
#ifdef MULTISAMPLING
Texture2DMS<float> gDepth : register(t2, space0);
#else
Texture2D<float> gDepth : register(t2, space0);
#endif
Texture2D<float4> gSceneColor : register(t3, space0);
[[vk::image_format("rgba16f")]] RWTexture2D<float4> gOutput : register(u4, space0);

float loadDepth(int2 pixel) {
#ifdef MULTISAMPLING
    return gDepth.Load(pixel, 0);
#else
    return gDepth.Load(int3(pixel, 0));
#endif
}

[numthreads(8, 8, 1)]
void CSMain(uint2 threadId : SV_DispatchThreadID) {
    if (any(threadId >= gConstants.outputSize)) {
        return;
    }

    const LightingParams params = gLightingParams[gConstants.sceneIndex];
    const int2 origin = int2(params.viewportRect.xy) + int2(threadId) * 4;
    const int2 end = int2(params.viewportRect.zw);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    [unroll]
    for (int y = 0; y < 4; y++) {
        [unroll]
        for (int x = 0; x < 4; x++) {
            const int2 pixel = origin + int2(x, y);
            if (all(pixel < end) && !lightingIsBackground(params, loadDepth(pixel))) {
                const float3 color = gSceneColor.Load(int3(pixel, 0)).rgb;
                sum += color * lightingEmissiveMask(color, params.emissiveParams.x);
            }
        }
    }

    gOutput[threadId] = float4(sum / 16.0f, 1.0f);
}
