//
// RT64
//
// Shared functions of the enhanced raster lighting passes that work on the depth buffer of a projection.
//

#pragma once

#include "shared/rt64_lighting_params.h"

// Position in world space of a pixel (centers at .5) with the given depth buffer value.
float3 lightingWorldPosition(LightingParams params, float2 pixel, float depth) {
    const float4 clipPosition = float4(pixel * params.pixelToClip.xy + params.pixelToClip.zw, depth * params.depthToClip.x + params.depthToClip.y, 1.0f);
    const float4 worldPosition = mul(params.invViewProj, clipPosition);
    return worldPosition.xyz / worldPosition.w;
}

// Clip space depth (z / w) of a depth buffer value, as used by the RSP for fog.
float lightingClipDepth(LightingParams params, float depth) {
    return depth * params.depthToClip.x + params.depthToClip.y;
}

// Fog the game applies at this depth, from 0 to 1.
float lightingFogAlpha(LightingParams params, float depth) {
    if (params.fog.z <= 0.0f) {
        return 0.0f;
    }

    const float clipDepth = max(lightingClipDepth(params, depth), 0.0f);
    return saturate((clipDepth * params.fog.x + params.fog.y) / 255.0f);
}

bool lightingIsBackground(LightingParams params, float depth) {
    return depth >= params.depthToClip.z;
}

// How much a color of a scene without a sun looks like a light (a lamp, a screen, a crystal): colorful and bright, or
// nearly white. Pale walls and floors (little color) and dark colorful surfaces (clothes, armor, dark stripes) don't
// glow.
float lightingEmissiveMask(float3 color, float threshold) {
    const float maxChannel = max(color.r, max(color.g, color.b));
    const float minChannel = min(color.r, min(color.g, color.b));
    const float luma = dot(color, float3(0.299f, 0.587f, 0.114f));
    const float colorful = smoothstep(0.3f, 0.55f, maxChannel - minChannel) * smoothstep(threshold, threshold + 0.12f, maxChannel) * smoothstep(0.3f, 0.4f, luma);
    const float white = smoothstep(0.9f, 0.98f, minChannel);
    return max(colorful, white);
}

// Octahedral encoding of unit vectors into two components in [0, 1].
float2 lightingOctWrap(float2 v) {
    return (1.0f - abs(v.yx)) * select(v.xy >= 0.0f, 1.0f, -1.0f);
}

float2 lightingEncodeNormal(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = (n.z >= 0.0f) ? n.xy : lightingOctWrap(n.xy);
    return n.xy * 0.5f + 0.5f;
}

float3 lightingDecodeNormal(float2 f) {
    f = f * 2.0f - 1.0f;
    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
    const float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0f, -t, t);
    return normalize(n);
}
