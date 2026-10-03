//
// RT64
//
// Lights the opaque surfaces of a projection after they're drawn: the sun with its shadow map, the light from the sky
// and the ground, and a light carried close to the camera in scenes without a sun. The game's colors already have
// their lighting baked in, so the result is a factor the color target is multiplied by (the pipeline blends with
// 2 * src * dst, so the factor is written halved and can brighten up to twice). Debug views replace the color instead.
//

#include "shared/rt64_lighting_params.h"

#include "Depth.hlsli"
#include "LightingCommon.hlsli"
#include "LightingSkyClouds.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingComposeCB> gConstants : register(b0, space0);
StructuredBuffer<LightingParams> gLightingParams : register(t1, space0);
// The parameters of this scene (the same as gLightingParams[sceneIndex]): read from a constant buffer, which mobile GPUs
// keep in their uniform memory. Copying the structure out of the storage buffer in every pixel was very slow there.
ConstantBuffer<LightingParams> gSceneParams : register(b11, space0);
#ifdef MULTISAMPLING
Texture2DMS<float> gDepth : register(t2, space0);
#else
Texture2D<float> gDepth : register(t2, space0);
#endif
Texture2D<float> gShadowMap : register(t3, space0);
SamplerComparisonState gShadowSampler : register(s4, space0);
Texture2D<float4> gNormalBuffer : register(t5, space0);
Texture2D<float4> gAmbientOcclusion : register(t6, space0);
StructuredBuffer<float4> gSkyAnalysis : register(t7, space0);

// Copy of the color target before the lighting and the light of its glowing surfaces at quarter resolution, only read
// in scenes with glowing surfaces (LightingParams::emissiveParams).
Texture2D<float4> gSceneColor : register(t8, space0);
Texture2D<float4> gEmissiveLight : register(t9, space0);

// Shadows of the light carried in scenes without a sun: the six faces of a cube around it in a 3x2 atlas.
Texture2D<float> gPointShadowMap : register(t10, space0);

// Tint of the light from the game's sky when it isn't a daytime sky (a sunset, a purple sky), from the average color the
// procedural sky measured on the previous frames.
float3 skyLightTint(LightingParams params) {
    if ((params.settings.w & LIGHTING_SCENE_FLAG_SKY_TINT) == 0) {
        return float3(1.0f, 1.0f, 1.0f);
    }

    const float3 skyColor = gSkyAnalysis[gConstants.sceneIndex % 8].rgb;
    const float luma = dot(skyColor, float3(0.299f, 0.587f, 0.114f));
    if (!(luma > 0.05f)) {
        return float3(1.0f, 1.0f, 1.0f);
    }

    const float daytime = smoothstep(0.02f, 0.10f, skyColor.b - skyColor.r) * smoothstep(-0.02f, 0.04f, skyColor.g - skyColor.r);
    const float3 chroma = saturate(skyColor / luma * 0.8f + 0.2f);
    return lerp(float3(1.0f, 1.0f, 1.0f), chroma, params.miscParams.x * (1.0f - daytime));
}

// Shadows of the clouds of the procedural sky: the shapes of its cloud layer (the same noise, wind and coverage), found
// where the ray from the position towards the sun crosses the layer. 1 is lit.
float cloudShadow(LightingParams params, float3 position) {
    // The entry of the analysis is only valid (and only read) when the CPU enabled the shadows. A NaN fails the
    // comparisons and turns them off too.
    float strength = params.cloudShadowParams.w;
    if (!(strength > 0.0f)) {
        return 1.0f;
    }

    if ((params.settings.w & LIGHTING_SCENE_FLAG_SKY_HIDDEN) == 0) {
        strength *= saturate(gSkyAnalysis[gConstants.sceneIndex % 8].w);
        if (!(strength > 0.0f)) {
            return 1.0f;
        }
    }

    // In sky space (y up) and cloud heights, from the origin of the world.
    const float3 relative = position - params.worldOrigin.xyz;
    const float3 skyPosition = float3(dot(relative, params.worldRight.xyz), dot(relative, params.worldUp.xyz), dot(relative, params.worldForward.xyz)) * params.cloudShadowParams.x;
    const float3 sun = float3(dot(params.sunDirection.xyz, params.worldRight.xyz), dot(params.sunDirection.xyz, params.worldUp.xyz), dot(params.sunDirection.xyz, params.worldForward.xyz));
    const float2 layerPosition = (skyPosition.xz + sun.xz / max(sun.y, 0.2f) * max(1.0f - skyPosition.y, 0.0f)) * params.cloudShadowParams.y;
    float2 shapePosition = layerPosition + params.cloudShadowOffset.xy;
    const uint octaves = uint(params.cloudShadowMisc.z);
    if (octaves > 0) {
        const float WarpFrequency = 0.25f;
        const float2 warpPosition = layerPosition * WarpFrequency + params.cloudShadowOffset.zw;
        const float warpX = lightingSkyFbm(warpPosition, 0.0f, 2, 101).x;
        const float warpY = lightingSkyFbm(warpPosition + float2(17.3f, 9.1f), 0.0f, 2, 117).x;
        shapePosition += (float2(warpX, warpY) - 0.5f) * params.cloudShadowMisc.x;
    }

    const float shape = lightingSkyFbm(shapePosition, 0.0f, max(octaves, 2u), 7).x;
    const float threshold = 0.60f - 0.28f * params.cloudShadowParams.z;
    const float softness = params.cloudShadowMisc.y;
    const float density = smoothstep(threshold - softness, threshold + softness, shape);
    return 1.0f - density * strength;
}

// Light cast by the glowing surfaces around the pixel (bilinear from the quarter resolution texture).
float3 sampleEmissiveLight(LightingParams params, float2 pixelPosition) {
    const float2 position = (pixelPosition - params.viewportRect.xy) * 0.25f - 0.5f;
    const int2 basePixel = int2(floor(position));
    const float2 fraction = position - float2(basePixel);
    const int2 maxPixel = max(int2(ceil((params.viewportRect.zw - params.viewportRect.xy) * 0.25f)) - 1, int2(0, 0));
    const float3 c00 = gEmissiveLight.Load(int3(clamp(basePixel, int2(0, 0), maxPixel), 0)).rgb;
    const float3 c10 = gEmissiveLight.Load(int3(clamp(basePixel + int2(1, 0), int2(0, 0), maxPixel), 0)).rgb;
    const float3 c01 = gEmissiveLight.Load(int3(clamp(basePixel + int2(0, 1), int2(0, 0), maxPixel), 0)).rgb;
    const float3 c11 = gEmissiveLight.Load(int3(clamp(basePixel + int2(1, 1), int2(0, 0), maxPixel), 0)).rgb;
    return lerp(lerp(c00, c10, fraction.x), lerp(c01, c11, fraction.x), fraction.y);
}

// Axes of a face of the point light's cube (the ones computePointShadowFace uses on the CPU) for a direction from the
// light: the face is the one of its largest component.
uint lightingPointShadowFace(float3 direction, out float3 forward, out float3 up) {
    const float3 magnitude = abs(direction);
    if ((magnitude.x >= magnitude.y) && (magnitude.x >= magnitude.z)) {
        forward = float3((direction.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
        up = float3(0.0f, 1.0f, 0.0f);
        return (direction.x >= 0.0f) ? 0 : 1;
    }
    else if (magnitude.y >= magnitude.z) {
        forward = float3(0.0f, (direction.y >= 0.0f) ? 1.0f : -1.0f, 0.0f);
        up = float3(0.0f, 0.0f, (direction.y >= 0.0f) ? -1.0f : 1.0f);
        return (direction.y >= 0.0f) ? 2 : 3;
    }
    else {
        forward = float3(0.0f, 0.0f, (direction.z >= 0.0f) ? 1.0f : -1.0f);
        up = float3(0.0f, 1.0f, 0.0f);
        return (direction.z >= 0.0f) ? 4 : 5;
    }
}

// Shadow of the light carried in scenes without a sun (1 is lit), with a 3x3 grid of comparisons that stays inside the
// face of the atlas.
float samplePointShadow(LightingParams params, float3 position, float3 normal) {
    const float faceSize = params.pointShadowParams.z;
    const float zNear = params.pointShadowPosition.w;
    float3 direction = position - params.pointShadowPosition.xyz;

    // A texel of a 90 degree face covers 2 * distance / size world units.
    const float texelSize = 2.0f * max(length(direction), zNear) / faceSize;
    direction += normal * (texelSize * params.pointShadowParams2.x);

    float3 forward, up;
    const uint face = lightingPointShadowFace(direction, forward, up);
    const float3 right = cross(up, forward);
    const float z = dot(direction, forward);
    if (z <= zNear) {
        return 1.0f;
    }

    const float2 faceUV = (float2(dot(direction, right), dot(direction, up)) / z) * float2(0.5f, -0.5f) + 0.5f;
    const float biasedZ = max(z - texelSize * params.pointShadowParams2.y, zNear);
    const float referenceDepth = params.pointShadowParams.x + params.pointShadowParams.y / biasedZ;
    const float2 tile = float2(float(face % 3), float(face / 3));
    const float2 atlasTexel = 1.0f / (float2(3.0f, 2.0f) * faceSize);
    const float2 minUV = (tile * faceSize + 1.0f) * atlasTexel;
    const float2 maxUV = ((tile + 1.0f) * faceSize - 1.0f) * atlasTexel;
    const float2 centerUV = (tile + faceUV) * faceSize * atlasTexel;
    const float2 spread = atlasTexel * params.pointShadowParams2.z;
    float lit = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; y++) {
        [unroll]
        for (int x = -1; x <= 1; x++) {
            const float2 sampleUV = clamp(centerUV + float2(x, y) * spread, minUV, maxUV);
            lit += gPointShadowMap.SampleCmpLevelZero(gShadowSampler, sampleUV, referenceDepth);
        }
    }

    return lerp(1.0f, lit / 9.0f, params.pointShadowParams.w);
}

float loadDepth(int2 pixel) {
#ifdef MULTISAMPLING
    return gDepth.Load(pixel, 0);
#else
    return gDepth.Load(int3(pixel, 0));
#endif
}

// Normal of the surface from the positions of the neighboring pixels, taking the side with the closest depth so the
// edges of objects don't bend it.
float3 normalFromDepth(LightingParams params, int2 pixel, float depth, float3 position) {
    const float depthLeft = loadDepth(pixel - int2(1, 0));
    const float depthRight = loadDepth(pixel + int2(1, 0));
    const float depthUp = loadDepth(pixel - int2(0, 1));
    const float depthDown = loadDepth(pixel + int2(0, 1));
    const bool useRight = abs(depthRight - depth) < abs(depthLeft - depth);
    const bool useDown = abs(depthDown - depth) < abs(depthUp - depth);
    const float2 center = float2(pixel) + 0.5f;
    const float3 positionX = useRight ? lightingWorldPosition(params, center + float2(1.0f, 0.0f), depthRight) : lightingWorldPosition(params, center - float2(1.0f, 0.0f), depthLeft);
    const float3 positionY = useDown ? lightingWorldPosition(params, center + float2(0.0f, 1.0f), depthDown) : lightingWorldPosition(params, center - float2(0.0f, 1.0f), depthUp);
    const float3 dX = (positionX - position) * (useRight ? 1.0f : -1.0f);
    const float3 dY = (positionY - position) * (useDown ? 1.0f : -1.0f);
    float3 normal = cross(dY, dX);
    const float normalLength = length(normal);
    normal = (normalLength > 1e-8f) ? (normal / normalLength) : -normalize(position - params.cameraPosition.xyz);

    // Face the camera.
    if (dot(normal, params.cameraPosition.xyz - position) < 0.0f) {
        normal = -normal;
    }

    return normal;
}

float sampleShadow(LightingParams params, float3 position, float3 normal, float NdotL, float foliageRadius) {
    // Move the position away from the surface to avoid self shadowing, more on surfaces at a grazing angle to the sun.
    // Foliage is moved out of its own tree towards the sun, so the crossed cards don't shadow each other but other
    // trees and objects still do.
    const float texelSize = params.shadowParams.x;
    const float slope = sqrt(saturate(1.0f - NdotL * NdotL));
    const float3 offsetPosition = position + normal * (texelSize * params.shadowParams.z * (0.5f + slope)) + params.sunDirection.xyz * (foliageRadius * params.foliageParams.w);
    const float4 shadowPosition = mul(params.shadowMatrix, float4(offsetPosition, 1.0f));
    const float3 shadowNdc = shadowPosition.xyz / shadowPosition.w;
    const float2 shadowUV = shadowNdc.xy * float2(0.5f, -0.5f) + 0.5f;
    if (any(shadowUV < 0.0f) || any(shadowUV > 1.0f) || (shadowNdc.z > 1.0f)) {
        return 1.0f;
    }

    // Grid of bilinear comparisons spread by the softness: a single one on low quality, 3x3 normally and 5x5 on ultra.
    const float referenceDepth = shadowNdc.z - params.shadowParams.y;
    const float2 spread = params.shadowMapParams.xy * params.shadowParams.w;
    const uint quality = params.settings.y;
    float lit = 0.0f;
    if (quality == 0) {
        lit = gShadowMap.SampleCmpLevelZero(gShadowSampler, shadowUV, referenceDepth);
    }
    else if (params.shadowParams2.x > 0.0f) {
        // Percentage closer soft shadows: the average depth of the casters found around the pixel gives how far they are
        // and so how wide the penumbra of the sun is there. Both passes take points of a golden angle spiral.
        const uint searchTaps = (quality >= 3) ? 16 : ((quality >= 2) ? 12 : 8);
        const uint filterTaps = (quality >= 3) ? 24 : ((quality >= 2) ? 16 : 12);
        const float2 mapSize = 1.0f / params.shadowMapParams.xy;
        const float maxSoftness = params.shadowParams2.z;
        float blockerSum = 0.0f;
        float blockerCount = 0.0f;
        for (uint i = 0; i < searchTaps; i++) {
            const float angle = float(i) * 2.39996323f;
            const float2 offset = float2(cos(angle), sin(angle)) * sqrt((float(i) + 0.5f) / float(searchTaps));
            const int2 texel = int2((shadowUV + offset * maxSoftness * params.shadowMapParams.xy) * mapSize);
            const float depth = gShadowMap.Load(int3(clamp(texel, int2(0, 0), int2(mapSize) - 1), 0));
            if (depth < referenceDepth) {
                blockerSum += depth;
                blockerCount += 1.0f;
            }
        }

        if (blockerCount <= 0.0f) {
            lit = 1.0f;
        }
        else {
            const float casterDistance = (referenceDepth - blockerSum / blockerCount) * params.shadowMapParams.w;
            const float softness = clamp(casterDistance * params.shadowParams2.x / texelSize, params.shadowParams2.y, maxSoftness);
            for (uint i = 0; i < filterTaps; i++) {
                const float angle = float(i) * 2.39996323f;
                const float2 offset = float2(cos(angle), sin(angle)) * sqrt((float(i) + 0.5f) / float(filterTaps));
                lit += gShadowMap.SampleCmpLevelZero(gShadowSampler, shadowUV + offset * softness * params.shadowMapParams.xy, referenceDepth);
            }

            lit /= float(filterTaps);
        }
    }
    else if (quality >= 3) {
        [unroll]
        for (int y = -2; y <= 2; y++) {
            [unroll]
            for (int x = -2; x <= 2; x++) {
                lit += gShadowMap.SampleCmpLevelZero(gShadowSampler, shadowUV + float2(x, y) * spread * 0.75f, referenceDepth);
            }
        }

        lit /= 25.0f;
    }
    else {
        [unroll]
        for (int y = -1; y <= 1; y++) {
            [unroll]
            for (int x = -1; x <= 1; x++) {
                lit += gShadowMap.SampleCmpLevelZero(gShadowSampler, shadowUV + float2(x, y) * spread, referenceDepth);
            }
        }

        lit /= 9.0f;
    }

    // Fade out the shadows towards the edge of the circle the shadow map covers around the player, the same in every
    // direction, so turning the camera doesn't change where they end.
    const float edgeFade = saturate((1.0f - length(shadowUV * 2.0f - 1.0f)) * 8.0f);
    return lerp(1.0f, lit, edgeFade * params.shadowMapParams.z);
}

// Ambient occlusion (x) and contact shadows (y) from the half resolution texture, taking the neighbors at the most
// similar distance to the camera.
float2 sampleAmbientOcclusion(LightingParams params, float2 pixelPosition, float cameraDistance) {
    if ((params.aoParams.w <= 0.0f) && (params.contactParams.w <= 0.0f)) {
        return float2(1.0f, 1.0f);
    }

    const float2 aoPosition = (pixelPosition - params.viewportRect.xy) * 0.5f - 0.5f;
    const int2 basePixel = int2(floor(aoPosition));
    const float2 fraction = aoPosition - float2(basePixel);
    const int2 maxPixel = int2(params.aoParams2.zw) - 1;
    float2 sum = float2(0.0f, 0.0f);
    float weightSum = 0.0f;
    [unroll]
    for (int y = 0; y < 2; y++) {
        [unroll]
        for (int x = 0; x < 2; x++) {
            const int2 samplePixel = clamp(basePixel + int2(x, y), int2(0, 0), maxPixel);
            const float4 value = gAmbientOcclusion.Load(int3(samplePixel, 0));
            const float bilinear = (x ? fraction.x : (1.0f - fraction.x)) * (y ? fraction.y : (1.0f - fraction.y));
            const float similarity = 1.0f / (abs(value.y - cameraDistance) / (cameraDistance * 0.02f + 1.0f) + 0.05f);
            const float weight = bilinear * similarity + 1e-5f;
            sum += value.xz * weight;
            weightSum += weight;
        }
    }

    return sum / weightSum;
}

float4 PSMain(in float4 pixelPosition : SV_POSITION
#ifdef MULTISAMPLING
    , out uint resultCoverage : SV_Coverage
#endif
    ) : SV_TARGET
{
    const LightingParams params = gSceneParams;
    const int2 pixel = int2(pixelPosition.xy);
    if (any(pixelPosition.xy < params.viewportRect.xy) || any(pixelPosition.xy >= params.viewportRect.zw)) {
        discard;
    }

#ifdef MULTISAMPLING
    // Edge pixels have samples of different surfaces. The nearest one is lit in the first pass and the farthest one in
    // the second, each writing only to its own samples.
    const uint sampleCount = params.settings.z;
    float nearestDepth = 1.0f;
    float farthestDepth = 0.0f;
    for (uint s = 0; s < sampleCount; s++) {
        const float sampleDepth = gDepth.Load(pixel, s);
        nearestDepth = min(nearestDepth, sampleDepth);
        farthestDepth = max(farthestDepth, sampleDepth);
    }

    const float tolerance = max(CoplanarDepthTolerance(nearestDepth) * 4.0f, 1e-5f);
    const bool edgePixel = (farthestDepth - nearestDepth) > tolerance;
    if ((gConstants.surfacePass == 1) && !edgePixel) {
        discard;
    }

    // Pass 2 is the single pass of the low quality preset: the nearest surface is lit for all the samples.
    const float depth = (gConstants.surfacePass == 1) ? farthestDepth : nearestDepth;
    resultCoverage = (gConstants.surfacePass == 2) ? ((1U << sampleCount) - 1U) : 0U;
    for (uint s = 0; (s < sampleCount) && (gConstants.surfacePass != 2); s++) {
        const float sampleDepth = gDepth.Load(pixel, s);
        const bool nearSurface = (abs(sampleDepth - nearestDepth) <= tolerance);
        const bool farSurface = (abs(sampleDepth - farthestDepth) <= tolerance);
        if ((gConstants.surfacePass == 0) ? nearSurface : (farSurface && !nearSurface)) {
            resultCoverage |= (1U << s);
        }
    }
#else
    const float depth = loadDepth(pixel);
#endif

    if (lightingIsBackground(params, depth)) {
        discard;
    }

    const float3 position = lightingWorldPosition(params, pixelPosition.xy, depth);
    // The normal buffer stores the nearest surface of each pixel. The farther surface of an edge pixel takes the normal
    // of a neighbor that shows it, so it's lit like the rest of that surface. So does a nearest surface the normal pass
    // left out, like the antialiased edge of a cutout whose coverage the raster pass kept but whose alpha the normal
    // pass rejected: the normal from the depth there mixes both surfaces and leaves dark lines along the edge.
    float4 normalSample = gNormalBuffer.Load(int3(pixel, 0));
#ifndef MULTISAMPLING
    const float tolerance = max(CoplanarDepthTolerance(depth) * 4.0f, 1e-5f);
#endif
    const bool farSurfacePass = (gConstants.surfacePass == 1);
    const bool gbufferScene = ((params.settings.w & LIGHTING_SCENE_FLAG_GBUFFER) != 0);
    if (farSurfacePass || ((normalSample.w <= 0.5f) && gbufferScene)) {
        const int2 Neighbors[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
        normalSample = float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (uint n = 0; n < 4; n++) {
            const int2 neighborPixel = pixel + Neighbors[n];
            const float neighborDepth = loadDepth(neighborPixel);
            const float4 neighborNormal = gNormalBuffer.Load(int3(neighborPixel, 0));
            if ((abs(neighborDepth - depth) <= tolerance * 4.0f) && (neighborNormal.w > 0.5f)) {
                normalSample = neighborNormal;
                break;
            }
        }
    }
#ifdef MULTISAMPLING
    if (farSurfacePass) {
        // A surface no neighbor shows is only seen through a crack between others (like the seams between the quads of a
        // wall): its normal can't be known, so it keeps its original color instead of being lit as something else.
        if ((normalSample.w <= 0.5f) && ((params.settings.w & LIGHTING_SCENE_FLAG_GBUFFER) != 0)) {
            discard;
        }
    }
#endif
    const bool storedNormal = (normalSample.w > 0.5f);
    const float3 normal = storedNormal ? lightingDecodeNormal(normalSample.xy) : normalFromDepth(params, pixel, depth, position);
    const float foliageRadius = storedNormal ? (normalSample.z * 4096.0f) : 0.0f;
    const bool foliage = (foliageRadius > 0.0f);
    const float3 worldUp = params.worldUp.xyz;

    // Light from the sky above and the ground below, occluded by the surroundings. The occlusion between the crossed
    // cards of foliage isn't real, so it's mostly left out there.
    const float2 occlusionSample = sampleAmbientOcclusion(params, pixelPosition.xy, length(params.cameraPosition.xyz - position));
    const float ambientOcclusion = (params.aoParams.w > 0.0f) ? occlusionSample.x : 1.0f;
    const float occlusion = lerp(1.0f, ambientOcclusion, params.aoParams.y * (foliage ? params.aoParams2.y : 1.0f));
    const float contact = (params.contactParams.w > 0.0f) ? lerp(1.0f, occlusionSample.y, params.contactParams.z) : 1.0f;
    const float upFactor = dot(normal, worldUp) * 0.5f + 0.5f;
    const float3 lightTint = skyLightTint(params);
    const float3 ambientLight = lerp(params.groundColor.rgb, params.ambientColor.rgb * lightTint, upFactor) * occlusion;

    // The sun.
    float3 sunLight = float3(0.0f, 0.0f, 0.0f);
    float shadow = 1.0f;
    if (params.sunDirection.w > 0.0f) {
        const float NdotL = dot(normal, params.sunDirection.xyz);
        const float wrap = foliage ? params.foliageParams.x : params.lightingParams.z;
        const float diffuse = saturate((NdotL + wrap) / (1.0f + wrap));
        shadow = min(sampleShadow(params, position, normal, NdotL, foliageRadius), contact);
        if (shadow > 0.0f) {
            shadow *= cloudShadow(params, position);
        }

        // Leaves are never completely dark, as light goes through them and bounces inside the canopy.
        if (foliage) {
            shadow = lerp(params.foliageParams.z, 1.0f, shadow);
        }

        // Surfaces facing away from the sun are already in shadow, so the shading from the normals is limited by the
        // strength set for it to keep the game's own shading readable.
        const float shading = lerp(1.0f, diffuse, params.lightingParams.w);
        sunLight = params.sunColor.rgb * lightTint * (shading * shadow);

        // Leaves glow when the sun is behind them.
        if (foliage) {
            const float3 viewDirection = normalize(position - params.cameraPosition.xyz);
            const float transmission = pow(saturate(dot(viewDirection, params.sunDirection.xyz)), 4.0f);
            sunLight += params.sunColor.rgb * (transmission * params.foliageParams.y * (0.35f + 0.65f * shadow));
        }
    }

    // Light carried with the camera.
    float3 pointLight = float3(0.0f, 0.0f, 0.0f);
    float pointShadow = 1.0f;
    if (params.pointLightPosition.w > 0.0f) {
        const float3 toLight = params.pointLightPosition.xyz - position;
        const float distance = length(toLight);
        const float3 lightDirection = toLight / max(distance, 1e-4f);
        const float attenuation = pow(saturate(1.0f - distance / params.pointLightPosition.w), max(params.pointLightColor.w, 0.5f));
        const float diffuse = saturate(dot(normal, lightDirection) * 0.75f + 0.25f);
        if ((params.pointShadowParams.w > 0.0f) && (attenuation > 0.0f)) {
            pointShadow = samplePointShadow(params, position, normal);
        }

        pointLight = params.pointLightColor.rgb * (diffuse * attenuation * pointShadow * ((params.sunDirection.w > 0.0f) ? 1.0f : contact));
    }

    // Glowing surfaces of scenes without a sun (lamps, screens, crystals) light their surroundings with their color and
    // stand out themselves, like the emissive surfaces of the path tracer.
    float3 emissiveLight = float3(0.0f, 0.0f, 0.0f);
    float selfEmission = 0.0f;
    if (params.emissiveParams.w > 0.0f) {
        const float3 glow = sampleEmissiveLight(params, pixelPosition.xy);
        emissiveLight = params.emissiveParams.w * (1.0f - exp(-glow * (params.emissiveParams.z / params.emissiveParams.w)));
        const float3 albedo = gSceneColor.Load(int3(pixel, 0)).rgb;
        selfEmission = lightingEmissiveMask(albedo, params.emissiveParams.x) * params.emissiveParams.y;
    }

    // Occlusion also darkens the direct light a bit, as the game's colors already include light from everywhere.
    const float directOcclusion = lerp(1.0f, occlusion, params.aoParams2.x);
    float3 factor = (ambientLight + (sunLight + pointLight + emissiveLight) * directOcclusion) * params.lightingParams.y + selfEmission;

    // The game's fog covers the lighting.
    const float fogAlpha = lightingFogAlpha(params, depth);
    factor = lerp(factor, float3(1.0f, 1.0f, 1.0f), fogAlpha);
    factor = lerp(float3(1.0f, 1.0f, 1.0f), factor, params.lightingParams.x);

    const uint debugView = params.settings.x;
    if (debugView == 1) {
        return float4(factor * 0.5f, 1.0f);
    }
    else if (debugView == 2) {
        return float4(normal * 0.5f + 0.5f, 1.0f);
    }
    else if (debugView == 3) {
        return float4(shadow.xxx, 1.0f);
    }
    else if (debugView == 4) {
        return float4(frac(position / 500.0f), 1.0f);
    }
    else if (debugView == 5) {
        return float4(fogAlpha.xxx, 1.0f);
    }
    else if (debugView == 6) {
        return float4(occlusion.xxx, 1.0f);
    }
    else if (debugView == 8) {
        return float4(contact.xxx, 1.0f);
    }
    else if (debugView == 10) {
        return float4(pointShadow.xxx, 1.0f);
    }
    else if (debugView == 11) {
        // The point light's atlas itself (one texel per pixel from the top left corner), darker where closer to it.
        const float storedDepth = gPointShadowMap.Load(int3(pixel, 0));
        return float4(saturate((1.0f - storedDepth) * 4.0f).xxx, 1.0f);
    }
    else if (debugView == 9) {
        // Light of the glowing surfaces, and red where a surface glows itself.
        return float4(saturate(emissiveLight + float3(selfEmission, 0.0f, 0.0f)), 1.0f);
    }
    else if (debugView == 7) {
        // Distance from the receiver to the occluder stored in the shadow map (red: occluder in front, green: behind),
        // and blue where the pixel has a stored normal.
        const float4 shadowPosition = mul(params.shadowMatrix, float4(position, 1.0f));
        const float3 shadowNdc = shadowPosition.xyz / shadowPosition.w;
        const float2 shadowUV = shadowNdc.xy * float2(0.5f, -0.5f) + 0.5f;
        uint shadowWidth, shadowHeight;
        gShadowMap.GetDimensions(shadowWidth, shadowHeight);
        const float storedDepth = gShadowMap.Load(int3(int2(shadowUV * float2(shadowWidth, shadowHeight)), 0));
        const float distance = (shadowNdc.z - storedDepth) * params.shadowMapParams.w;
        return float4(saturate(distance / 200.0f), saturate(-distance / 200.0f), storedNormal ? 1.0f : 0.0f, 1.0f);
    }

    return float4(saturate(factor * 0.5f), 1.0f);
}
