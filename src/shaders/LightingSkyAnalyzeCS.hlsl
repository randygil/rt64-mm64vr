//
// RT64
//
// Decides whether the game's sky behind a scene looks like a daytime sky the procedural sky can replace. The average
// color of the background above the horizon must be blue, and nearly all of it must be keyed as sky: sunsets, night
// skies and skies of other colors (purple, mint green) would otherwise become a mismatched blue sky with patches of the
// original wherever it isn't keyed. The estimate is smoothed over the frames and kept while too little of the sky is
// visible (looking down).
//
// Output (one entry per scene): xyz average color of the sky, w how much of it can be replaced (0 to 1).
//

#include "LightingSkyCommon.hlsli"
#include "LightingSkyClouds.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingSkyAnalysisCB> gConstants : register(b0, space0);
StructuredBuffer<LightingSkyParams> gSkyParams : register(t1, space0);
#ifdef MULTISAMPLING
Texture2DMS<float> gDepth : register(t2, space0);
#else
Texture2D<float> gDepth : register(t2, space0);
#endif
Texture2D<float4> gSceneColor : register(t3, space0);
RWStructuredBuffer<float4> gAnalysis : register(u4, space0);

#define GRID_SIZE 32
#define THREAD_COUNT 64

groupshared float4 gSums[THREAD_COUNT];
groupshared float gKeySums[THREAD_COUNT];

float loadDepth(int2 pixel) {
#ifdef MULTISAMPLING
    return gDepth.Load(pixel, 0);
#else
    return gDepth.Load(int3(pixel, 0));
#endif
}

[numthreads(THREAD_COUNT, 1, 1)]
void CSMain(uint threadIndex : SV_GroupIndex) {
    const LightingSkyParams params = gSkyParams[gConstants.slot];
    const float2 rectMin = params.rect.xy;
    const float2 rectSize = params.rect.zw - params.rect.xy;
    const float threshold = params.rayOrigin.w;
    const uint samplesPerThread = (GRID_SIZE * GRID_SIZE) / THREAD_COUNT;
    float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float keySum = 0.0f;
    for (uint i = 0; i < samplesPerThread; i++) {
        const uint sampleIndex = threadIndex * samplesPerThread + i;
        const float2 gridPosition = (float2(sampleIndex % GRID_SIZE, sampleIndex / GRID_SIZE) + 0.5f) / float(GRID_SIZE);
        const float2 pixelPosition = rectMin + gridPosition * rectSize;
        const int2 pixel = int2(pixelPosition);
        if (loadDepth(pixel) < threshold) {
            continue;
        }

        // Only well above the horizon, where the game's sky isn't blended with the fog or the scenery.
        const float3 direction = normalize(params.rayOrigin.xyz + pixelPosition.x * params.rayStepX.xyz + pixelPosition.y * params.rayStepY.xyz);
        if (direction.y < 0.08f) {
            continue;
        }

        const float3 color = gSceneColor.Load(int3(pixel, 0)).rgb;
        sum += float4(color, 1.0f);
        keySum += lightingSkyKey(color, params.blendParams.z, params.outputParams.y);
    }

    gSums[threadIndex] = sum;
    gKeySums[threadIndex] = keySum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = THREAD_COUNT / 2; stride > 0; stride /= 2) {
        if (threadIndex < stride) {
            gSums[threadIndex] += gSums[threadIndex + stride];
            gKeySums[threadIndex] += gKeySums[threadIndex + stride];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (threadIndex != 0) {
        return;
    }

    const float4 total = gSums[0];
    const float4 previous = gAnalysis[gConstants.sceneIndex];
    if (total.w < 24.0f) {
        // Not enough sky to judge it: keep what was seen before, or allow the replacement until there's some.
        gAnalysis[gConstants.sceneIndex] = (gConstants.reset != 0) ? float4(0.0f, 0.0f, 0.0f, 1.0f) : previous;
        return;
    }

    // Daytime skies have more blue than red, and more green than red. Purple skies and sunsets have more red than green.
    const float3 mean = total.rgb / total.w;
    const float luma = dot(mean, float3(0.299f, 0.587f, 0.114f));
    const float daytime = smoothstep(0.02f, 0.10f, mean.b - mean.r) * smoothstep(-0.02f, 0.04f, mean.g - mean.r) * smoothstep(0.2f, 0.35f, luma);

    // A sky only partly keyed would be replaced in patches.
    const float keyed = gKeySums[0] / total.w;
    const float4 current = float4(mean, daytime * smoothstep(0.6f, 0.85f, keyed));
    gAnalysis[gConstants.sceneIndex] = (gConstants.reset != 0) ? current : lerp(previous, current, 0.1f);
}
