//
// RT64
//
// Procedural sky of the enhanced lighting. The pixels of the background (where nothing was drawn) that look like sky
// in the game's picture (blue, or bright and neutral like painted clouds) above the horizon get an atmosphere from the
// sky-view table, the sun and a layer of clouds anchored to the world. Close to the horizon the sky fades into the
// game's own colors so the fog and the 2D scenery still match. The key is the alpha of the blend, so the edges against
// the painted scenery stay soft, and multisampled targets only write the samples of the background.
//

#include "Color.hlsli"
#include "LightingSkyCommon.hlsli"
#include "LightingSkyClouds.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingSkyCB> gConstants : register(b0, space0);
StructuredBuffer<LightingSkyParams> gSkyParams : register(t1, space0);
#ifdef MULTISAMPLING
Texture2DMS<float> gDepth : register(t2, space0);
#else
Texture2D<float> gDepth : register(t2, space0);
#endif
Texture2D<float4> gSceneColor : register(t3, space0);
Texture2D<float4> gSkyLut : register(t4, space0);
SamplerState gLinearSampler : register(s5, space0);
StructuredBuffer<float4> gSkyAnalysis : register(t6, space0);

float3 sampleSkyLut(float2 coordinates) {
    return gSkyLut.SampleLevel(gLinearSampler, coordinates, 0.0f).rgb;
}

float2 skyLutTexel(float u, float v) {
    const float2 size = float2(LIGHTING_SKY_LUT_WIDTH, LIGHTING_SKY_LUT_HEIGHT);
    return (float2(u, v) * (size - 1.0f) + 0.5f) / size;
}

#ifdef MULTISAMPLING
// Bit mask of the samples of a pixel that are background.
uint backgroundMask(int2 pixel, uint sampleCount, float threshold) {
    uint mask = 0;
    for (uint i = 0; i < sampleCount; i++) {
        if (gDepth.Load(pixel, i) >= threshold) {
            mask |= (1u << i);
        }
    }

    return mask;
}
#endif

float4 PSMain(in float4 pixelPosition : SV_POSITION
#ifdef MULTISAMPLING
    , out uint coverage : SV_Coverage
#endif
    ) : SV_TARGET
{
    const LightingSkyParams params = gSkyParams[gConstants.slot];
    const int2 pixel = int2(pixelPosition.xy);
    const float threshold = params.rayOrigin.w;
#ifdef MULTISAMPLING
    const uint sampleCount = clamp(params.settings.z, 1u, 16u);
    const uint fullMask = (1u << sampleCount) - 1u;
    const uint mask = backgroundMask(pixel, sampleCount, threshold);
    if (mask == 0) {
        discard;
    }

    coverage = mask;
    float3 original = gSceneColor.Load(int3(pixel, 0)).rgb;

    // The resolved color of a pixel on an edge of the scene mixes it with the background, so the background is taken
    // from the neighbors that are only background instead.
    if (mask != fullMask) {
        const int2 rectMin = int2(params.rect.xy);
        const int2 rectMax = int2(params.rect.zw) - 1;
        float3 neighborSum = float3(0.0f, 0.0f, 0.0f);
        float neighborCount = 0.0f;
        [unroll]
        for (uint i = 0; i < 4; i++) {
            const int2 offset = (i == 0) ? int2(-1, 0) : ((i == 1) ? int2(1, 0) : ((i == 2) ? int2(0, -1) : int2(0, 1)));
            const int2 neighbor = clamp(pixel + offset, rectMin, rectMax);
            if (backgroundMask(neighbor, sampleCount, threshold) == fullMask) {
                neighborSum += gSceneColor.Load(int3(neighbor, 0)).rgb;
                neighborCount += 1.0f;
            }
        }

        if (neighborCount > 0.0f) {
            original = neighborSum / neighborCount;
        }
    }
#else
    if (gDepth.Load(int3(pixel, 0)) < threshold) {
        discard;
    }

    const float3 original = gSceneColor.Load(int3(pixel, 0)).rgb;
#endif

    const float3 direction = normalize(params.rayOrigin.xyz + pixelPosition.x * params.rayStepX.xyz + pixelPosition.y * params.rayStepY.xyz);
    const uint debugView = params.settings.y;
    const bool replaceAll = ((params.settings.w & LIGHTING_SKY_FLAG_REPLACE_ALL) != 0) || (debugView == 2);
    // Only skies that look like daytime skies are replaced (see LightingSkyAnalyzeCS).
    float key = replaceAll ? 1.0f : (lightingSkyKey(original, params.blendParams.z, params.outputParams.y) * gSkyAnalysis[gConstants.sceneIndex].w);
    key *= smoothstep(-0.02f, 0.06f, direction.y);
    if (debugView == 1) {
        return float4(key, key, key, 1.0f);
    }

    if (key < 0.002f) {
        discard;
    }

    const float3 sunDirection = params.sunDirection.xyz;
    const float pixelAngle = params.rayStepX.w;
    float3 sky = sampleSkyLut(lightingSkyLutCoordinates(direction, sunDirection));
    sky += params.sunColor.rgb * lightingSkySun(direction, sunDirection, params.sunDirection.w, pixelAngle, params.sunColor.w, params.groundColor.w);

    if ((direction.y > 0.0f) && (params.cloudParams.y > 0.0f) && (debugView != 4)) {
        // The light of the sky on the clouds comes from fixed directions of the table: the zenith and the horizon
        // towards and away from the sun (at an elevation of asin(0.1)).
        const float HorizonRow = 0.2525f;
        const float3 zenith = sampleSkyLut(skyLutTexel(0.5f, 1.0f));
        const float3 horizonSun = sampleSkyLut(skyLutTexel(0.0f, HorizonRow));
        const float3 horizonAway = sampleSkyLut(skyLutTexel(1.0f, HorizonRow));
        const uint quality = min(params.settings.x, 3u);

        LightingSkyCloudLayer layer;
        layer.origin = params.cloudOrigin.xy;
        layer.frequency = params.cloudOrigin.z;
        layer.curvature = params.cloudOrigin.w;
        layer.offset = params.cloudOffset;
        layer.coverage = params.cloudParams.x;
        layer.opacity = params.cloudParams.y;
        layer.warp = params.cloudParams.z;
        layer.absorption = params.cloudParams.w;
        layer.sunStrength = params.cloudLight.x;
        layer.ambientStrength = params.cloudLight.y;
        layer.belly = params.cloudLight.z;
        layer.haze = params.cloudLight.w;
        layer.sunColor = params.sunColor.rgb;
        layer.groundColor = params.groundColor.rgb;
        layer.ambientTop = zenith * 0.5f + (horizonSun + horizonAway) * 0.25f;
        layer.ambientHorizon = (horizonSun + horizonAway) * 0.5f;

        // Low skips the warp, so its clouds drift without changing shape.
        layer.octaves = 3 + quality;
        layer.warpOctaves = min(quality, 2u);
        layer.lightTaps = 1 + quality;

        const float4 clouds = lightingSkyClouds(layer, direction, sunDirection, pixelAngle);
        if (debugView == 3) {
            return float4(clouds.aaa, 1.0f);
        }

        sky = lerp(sky, clouds.rgb, clouds.a);
    }

    // Close to the horizon the sky fades into the game's own colors.
    const float fade = 1.0f - saturate(direction.y / max(params.blendParams.y, 1e-3f));
    sky = lerp(sky, SrgbToLinear(original), params.blendParams.x * fade * fade);

    float3 color = LinearToSrgb(lightingSkyCompressHighlights(sky, params.blendParams.w));
    color += lightingSkyDither(pixelPosition.xy) * params.outputParams.x;
    return float4(color, key);
}
