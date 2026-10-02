//
// RT64
//
// Parameters of the procedural sky of the enhanced lighting, shared by the CPU (render/rt64_lighting_sky.cpp) and its
// shaders. Directions are expressed in sky space: the basis of the world given by the host (x right, y up, z forward),
// so the sky doesn't need to know anything about the game.
//

#pragma once

#include "shared/rt64_hlsl.h"

// Size of the sky-view lookup table: relative azimuth to the sun (0 to 180 degrees) by elevation (0 to 90 degrees).
#define LIGHTING_SKY_LUT_WIDTH      64
#define LIGHTING_SKY_LUT_HEIGHT     32

// Flags of LightingSkyParams::settings.w.
#define LIGHTING_SKY_FLAG_REPLACE_ALL   0x1

#ifdef HLSL_CPU
namespace interop {
#endif
    struct LightingSkyParams {
        // Direction of the view ray (not normalized) through the pixel at (0, 0), with pixel centers at .5, and its change
        // per pixel. The direction of any pixel is rayOrigin + x * rayStepX + y * rayStepY. W components: the depth
        // from which a pixel is background, the angular size of a pixel in radians and nothing.
        float4 rayOrigin;
        float4 rayStepX;
        float4 rayStepY;

        // Region of the target in pixels (left, top, right, bottom).
        float4 rect;

        // xyz direction towards the sun, w angular radius of the sun disc.
        float4 sunDirection;

        // Sunlight that reaches the ground and the clouds, w strength of the sun disc.
        float4 sunColor;

        // Light bounced by the ground onto the bottom of the clouds, w strength of the glow around the sun.
        float4 groundColor;

        // xy position of the camera under the cloud layer in cloud heights, z cells of the cloud noise per cloud height
        // and w radius of curvature of the layer in cloud heights.
        float4 cloudOrigin;

        // xy offset of the shape of the clouds and zw offset of the low frequency noise that warps it, in cells of each
        // noise. They move with the wind and wrap at the period of the noise.
        float4 cloudOffset;

        // x coverage, y opacity, z warp, w absorption of the sunlight inside the clouds.
        float4 cloudParams;

        // x strength of the sunlight, y strength of the light of the sky, z darkening of the bottom of thick clouds and
        // w haze of distant clouds.
        float4 cloudLight;

        // x amount of the game's own colors kept close to the horizon, y height of that blend (sine of the elevation),
        // z keying of bright neutral pixels (painted clouds) as sky and w start of the highlight compression.
        float4 blendParams;

        // x amplitude of the dither in output units, y luminance under which the background isn't keyed as sky.
        float4 outputParams;

        // x quality (0 low to 3 ultra), y debug view, z sample count of the depth buffer and w flags.
        uint4 settings;
    };

    struct LightingSkyCB {
        uint slot;
        uint3 padding;
    };

    struct LightingSkyLutCB {
        // xyz direction towards the sun in the frame of the table (the sun has azimuth 0), w exposure.
        float4 sunDirection;

        // rgb tint applied to the sky, w saturation.
        float4 skyTint;

        // x density of the aerosols (Mie haze), y multiple scattering, z ozone, w exponent that compresses the range
        // between the dark zenith and the bright horizon (1 keeps it).
        float4 atmosphere;

        // x altitude of the observer in meters.
        float4 observer;
    };
#ifdef HLSL_CPU
};
#endif

#ifndef HLSL_CPU
#define LIGHTING_SKY_PI 3.14159265f

// Direction of a texel of the sky-view table. The sun is at azimuth 0 along +x. Columns cover the azimuth relative to
// the sun linearly and rows the elevation with a square law, so the horizon gets more resolution than the zenith.
float3 lightingSkyLutDirection(uint2 texel) {
    const float u = float(texel.x) / float(LIGHTING_SKY_LUT_WIDTH - 1);
    const float v = float(texel.y) / float(LIGHTING_SKY_LUT_HEIGHT - 1);
    const float azimuth = u * LIGHTING_SKY_PI;
    const float elevation = v * v * (LIGHTING_SKY_PI * 0.5f);
    return float3(cos(elevation) * cos(azimuth), sin(elevation), cos(elevation) * sin(azimuth));
}

// Coordinates of the sky-view table for a direction above the horizon, with texel centers at the ends of the range.
float2 lightingSkyLutCoordinates(float3 direction, float3 sunDirection) {
    const float2 horizontal = direction.xz;
    const float2 sunHorizontal = sunDirection.xz;
    const float horizontalLength = length(horizontal);
    const float sunHorizontalLength = length(sunHorizontal);
    float cosine = 1.0f;
    if ((horizontalLength > 1e-5f) && (sunHorizontalLength > 1e-5f)) {
        cosine = dot(horizontal, sunHorizontal) / (horizontalLength * sunHorizontalLength);
    }

    const float u = acos(clamp(cosine, -1.0f, 1.0f)) / LIGHTING_SKY_PI;
    const float v = sqrt(asin(saturate(direction.y)) / (LIGHTING_SKY_PI * 0.5f));
    const float2 size = float2(LIGHTING_SKY_LUT_WIDTH, LIGHTING_SKY_LUT_HEIGHT);
    return (float2(u, v) * (size - 1.0f) + 0.5f) / size;
}
#endif
