//
// RT64
//
// Ambient occlusion of the opaque surfaces of a projection at half resolution, with the horizon based integral of
// Ground Truth Ambient Occlusion (Jimenez et al. 2016): for a few directions around each pixel, the highest horizon on
// both sides is searched in the depth buffer within a radius in world units, and the visible arc is integrated with
// the normal projected onto that slice. Directions and steps are rotated per pixel and smoothed afterwards by the blur.
//
// The same pass marches the contact shadows: a short ray from each pixel towards the light (the sun, or the light
// carried with the camera in scenes without one) through the depth buffer. The pixel is in shadow if the ray goes behind
// a visible surface that isn't much thicker than the ray's distance to it, which catches the fine detail the shadow map
// misses (where objects touch the ground) and gives the only shadows of the light carried with the camera.
//
// Output: x occlusion (1 = unoccluded), y distance of the pixel to the camera (for the depth aware filters), z light
// reaching the pixel past the contact shadows (1 = lit).
//

#include "shared/rt64_lighting_params.h"

#include "LightingCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingAOCB> gConstants : register(b0, space0);
StructuredBuffer<LightingParams> gLightingParams : register(t1, space0);
#ifdef MULTISAMPLING
Texture2DMS<float> gDepth : register(t2, space0);
#else
Texture2D<float> gDepth : register(t2, space0);
#endif
Texture2D<float4> gNormalBuffer : register(t3, space0);
[[vk::image_format("rgba16f")]] RWTexture2D<float4> gOutput : register(u4, space0);

static const float Pi = 3.14159265f;

float loadDepth(int2 pixel) {
#ifdef MULTISAMPLING
    return gDepth.Load(pixel, 0);
#else
    return gDepth.Load(int3(pixel, 0));
#endif
}

// Pixel of the framebuffer where a world position is projected to.
float2 worldToPixel(LightingParams params, float3 position) {
    const float4 clipPosition = mul(params.viewProj, float4(position, 1.0f));
    const float2 clipXY = clipPosition.xy / clipPosition.w;
    return (clipXY - params.pixelToClip.zw) / params.pixelToClip.xy;
}

float3 normalAt(LightingParams params, int2 pixel, float depth, float3 position) {
    const float4 normalSample = gNormalBuffer.Load(int3(pixel, 0));
    if (normalSample.w > 0.5f) {
        return lightingDecodeNormal(normalSample.xy);
    }

    // Fall back to the normal of the face from the neighboring depths.
    const float2 center = float2(pixel) + 0.5f;
    const float3 positionX = lightingWorldPosition(params, center + float2(1.0f, 0.0f), loadDepth(pixel + int2(1, 0)));
    const float3 positionY = lightingWorldPosition(params, center + float2(0.0f, 1.0f), loadDepth(pixel + int2(0, 1)));
    float3 normal = cross(positionY - position, positionX - position);
    normal = (dot(normal, normal) > 1e-12f) ? normalize(normal) : normalize(params.cameraPosition.xyz - position);
    return (dot(normal, params.cameraPosition.xyz - position) < 0.0f) ? -normal : normal;
}

// Interleaved gradient noise (Jimenez 2014), a cheap per pixel rotation that the blur turns into a smooth result. It
// doesn't change between frames: there's no temporal accumulation to average it, so it would flicker.
float gradientNoise(float2 pixel) {
    return frac(52.9829189f * frac(dot(pixel, float2(0.06711056f, 0.00583715f))));
}

float contactShadow(LightingParams params, int2 pixel, float3 position, float3 normal, float cameraDistance, float noise) {
    const uint stepCount = uint(params.contactParams.w);
    if (stepCount == 0) {
        return 1.0f;
    }

    float3 lightDirection;
    float rayLength = params.contactParams.x;
    if (params.sunDirection.w > 0.0f) {
        lightDirection = params.sunDirection.xyz;
    }
    else if (params.pointLightPosition.w > 0.0f) {
        const float3 toLight = params.pointLightPosition.xyz - position;
        const float lightDistance = length(toLight);
        lightDirection = toLight / max(lightDistance, 1e-4f);
        rayLength = min(rayLength, lightDistance * 0.9f);
    }
    else {
        return 1.0f;
    }

    // Surfaces facing away from the light are already dark from their shading. Light grazing a surface would stretch
    // the shadows of its smallest details (like a sign on a wall) into long streaks, so they fade out there.
    const float NdotL = dot(normal, lightDirection);
    if (NdotL <= 0.0f) {
        return 1.0f;
    }

    float strength = saturate(NdotL * 4.0f);

    // The light carried with the camera sits right in front of it, so it would throw long streaks across the walls next
    // to the camera, which are seen nearly edge on and fool the depth buffer test. Its contact shadows are kept for the
    // floors, where they ground the characters.
    if (params.sunDirection.w <= 0.0f) {
        strength *= saturate(dot(normal, params.worldUp.xyz) * 2.5f - 1.25f);
        if (strength <= 0.0f) {
            return 1.0f;
        }
    }

    // Rays covering only a couple of pixels on the screen can't find anything.
    const float3 origin = position + normal * (1.0f + cameraDistance * 0.002f);
    const float4 startClip = mul(params.viewProj, float4(origin, 1.0f));
    const float4 endClip = mul(params.viewProj, float4(origin + lightDirection * rayLength, 1.0f));
    if ((startClip.w <= 1e-3f) || (endClip.w <= 1e-3f)) {
        return 1.0f;
    }

    const float2 startPixel = (startClip.xy / startClip.w - params.pixelToClip.zw) / params.pixelToClip.xy;
    const float2 endPixel = (endClip.xy / endClip.w - params.pixelToClip.zw) / params.pixelToClip.xy;
    if (length(endPixel - startPixel) < 3.0f) {
        return 1.0f;
    }

    // Surfaces are given a thickness that grows with the distance, as the depth buffer only has their front.
    const float thickness = params.contactParams.y * (1.0f + cameraDistance / 2000.0f);
    for (uint s = 0; s < stepCount; s++) {
        const float t = (float(s) + noise) / float(stepCount);
        const float4 rayClip = mul(params.viewProj, float4(origin + lightDirection * (t * rayLength), 1.0f));
        if (rayClip.w <= 1e-3f) {
            break;
        }

        const float2 rayPixel = (rayClip.xy / rayClip.w - params.pixelToClip.zw) / params.pixelToClip.xy;
        if (any(rayPixel < params.viewportRect.xy) || any(rayPixel >= params.viewportRect.zw)) {
            break;
        }

        const int2 samplePixel = int2(rayPixel);
        if (all(samplePixel == pixel)) {
            continue;
        }

        const float sampleDepth = loadDepth(samplePixel);
        if (lightingIsBackground(params, sampleDepth)) {
            continue;
        }

        // Compared as distances along the view direction (the w of the clip space).
        const float3 samplePosition = lightingWorldPosition(params, float2(samplePixel) + 0.5f, sampleDepth);
        const float sampleW = mul(params.viewProj, float4(samplePosition, 1.0f)).w;
        // Layers just in front of a surface (signs, posters, decals drawn as their own geometry) don't count.
        const float behind = rayClip.w - sampleW;
        if ((behind > max(4.0f, rayClip.w * 0.004f)) && (behind < thickness)) {
            // Occluders found towards the end of the ray cast a lighter shadow, so the shadows fade out at their tips.
            return lerp(1.0f, smoothstep(0.5f, 1.0f, t), strength);
        }
    }

    return 1.0f;
}

[numthreads(8, 8, 1)]
void CSMain(uint2 threadId : SV_DispatchThreadID) {
    if (any(threadId >= gConstants.outputSize)) {
        return;
    }

    const LightingParams params = gLightingParams[gConstants.sceneIndex];
    const int2 pixel = int2(params.viewportRect.xy) + int2(threadId * 2);
    const float depth = loadDepth(pixel);
    if (lightingIsBackground(params, depth) || any(float2(pixel) >= params.viewportRect.zw)) {
        gOutput[threadId] = float4(1.0f, 65000.0f, 1.0f, 1.0f);
        return;
    }

    const float2 pixelCenter = float2(pixel) + 0.5f;
    const float3 position = lightingWorldPosition(params, pixelCenter, depth);
    const float3 normal = normalAt(params, pixel, depth, position);
    const float3 viewVector = normalize(params.cameraPosition.xyz - position);
    const float cameraDistance = length(params.cameraPosition.xyz - position);
    const float noise = gradientNoise(float2(threadId));

    // Foliage cards are flat, so they would shadow themselves.
    const bool foliage = (gNormalBuffer.Load(int3(pixel, 0)).z > 0.0f);
    const float contact = foliage ? 1.0f : contactShadow(params, pixel, position, normal, cameraDistance, frac(noise * 3.71f + 0.13f));

    // Radius of the search on the screen, from the radius in the world at the distance of the pixel.
    const uint sliceCount = uint(params.aoParams.w);
    const float worldRadius = params.aoParams.x;
    const float3 sideDirection = normalize(cross(viewVector, abs(viewVector.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f)));
    const float pixelRadius = length(worldToPixel(params, position + sideDirection * worldRadius) - pixelCenter);
    if ((sliceCount == 0) || (pixelRadius < 1.0f)) {
        gOutput[threadId] = float4(1.0f, cameraDistance, contact, 1.0f);
        return;
    }

    const uint stepCount = 6;
    const float maxPixelRadius = min(pixelRadius, 256.0f);
    const float stepNoise = frac(noise * 7.13f + 0.37f);
    float visibility = 0.0f;
    for (uint slice = 0; slice < sliceCount; slice++) {
        const float angle = (float(slice) + noise) * (Pi / float(sliceCount));
        const float2 direction = float2(cos(angle), sin(angle));

        // Slice plane in world space: it contains the view vector and the screen direction.
        const float3 slicePoint = lightingWorldPosition(params, pixelCenter + direction * 4.0f, depth);
        const float3 sliceTangent = normalize(slicePoint - position - viewVector * dot(slicePoint - position, viewVector));
        const float3 sliceNormal = normalize(cross(sliceTangent, viewVector));
        const float3 projectedNormal = normal - sliceNormal * dot(normal, sliceNormal);
        const float projectedLength = length(projectedNormal);
        if (projectedLength < 1e-4f) {
            visibility += 1.0f;
            continue;
        }

        const float cosNormal = saturate(dot(projectedNormal / projectedLength, viewVector));
        const float normalAngle = sign(dot(projectedNormal, sliceTangent)) * acos(cosNormal);

        // Highest horizon on each side, as the cosine of its angle to the view vector.
        float horizonCos[2] = { -1.0f, -1.0f };
        for (uint side = 0; side < 2; side++) {
            const float2 sideDirection2D = (side == 0) ? direction : -direction;
            for (uint s = 0; s < stepCount; s++) {
                const float t = (float(s) + stepNoise) / float(stepCount);
                const float2 samplePixel = pixelCenter + sideDirection2D * (1.0f + t * t * maxPixelRadius);
                if (any(samplePixel < params.viewportRect.xy) || any(samplePixel >= params.viewportRect.zw)) {
                    break;
                }

                const float sampleDepth = loadDepth(int2(samplePixel));
                if (lightingIsBackground(params, sampleDepth)) {
                    continue;
                }

                const float3 samplePosition = lightingWorldPosition(params, samplePixel, sampleDepth);
                const float3 delta = samplePosition - position;

                // Layers just above the surface (signs, posters, decals drawn as their own geometry) don't occlude it. Seen at
                // a grazing angle, their edges would darken the wall next to them.
                if (dot(delta, normal) < params.miscParams.y) {
                    continue;
                }

                const float distance = length(delta);
                const float sampleCos = dot(delta / max(distance, 1e-4f), viewVector);

                // Occluders fade out towards the edge of the radius so the result doesn't depend on hard cut offs.
                const float falloff = saturate((worldRadius - distance) / (worldRadius * 0.4f));
                horizonCos[side] = max(horizonCos[side], lerp(-1.0f, sampleCos, falloff));
            }
        }

        // Integrate the visible arc between both horizons, weighted by the cosine with the normal (GTAO eq. 10).
        const float h0 = normalAngle + max(-acos(horizonCos[1]) - normalAngle, -Pi * 0.5f);
        const float h1 = normalAngle + min(acos(horizonCos[0]) - normalAngle, Pi * 0.5f);
        const float sinNormal = sin(normalAngle);
        const float arc0 = -cos(2.0f * h0 - normalAngle) + cosNormal + 2.0f * h0 * sinNormal;
        const float arc1 = -cos(2.0f * h1 - normalAngle) + cosNormal + 2.0f * h1 * sinNormal;
        visibility += projectedLength * 0.25f * (arc0 + arc1);
    }

    visibility = saturate(visibility / float(sliceCount));
    gOutput[threadId] = float4(pow(visibility, params.aoParams.z), cameraDistance, contact, 1.0f);
}
