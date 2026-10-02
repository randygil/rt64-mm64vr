//
// RT64
//
// Functions of the procedural sky that only depend on their inputs (no resources), so other renderers can reuse them:
// a periodic gradient noise, a filtered fractal sum of it, a 2.5D layer of clouds lit by the sun, the sun disc, the key
// that tells sky pixels of a background picture apart from painted scenery, a highlight compression and a dither.
//

#pragma once

uint lightingSkyHash(uint2 q) {
    q *= uint2(1597334673u, 3812015801u);
    return (q.x ^ q.y) * 1597334673u;
}

float lightingSkyGradient(uint hash, float2 offset) {
    const float2 gradient = float2(float(hash & 0xFFFFu), float(hash >> 16)) * (1.0f / 32767.5f) - 1.0f;
    return dot(gradient, offset) * rsqrt(max(dot(gradient, gradient), 1e-4f));
}

// Gradient noise in [0, 1] with a mean of 0.5. It repeats every 256 cells, so offsets that wrap at 256 don't cause seams.
float lightingSkyNoise(float2 p, uint seed) {
    const float2 cell = floor(p);
    const float2 f = p - cell;
    const float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    const uint2 q0 = asuint(int2(cell)) & 255u;
    const uint2 q1 = (q0 + 1u) & 255u;
    const uint2 s = uint2(seed * 263u + 71u, seed * 431u + 29u);
    const float a = lightingSkyGradient(lightingSkyHash(uint2(q0.x, q0.y) + s), f);
    const float b = lightingSkyGradient(lightingSkyHash(uint2(q1.x, q0.y) + s), f - float2(1.0f, 0.0f));
    const float c = lightingSkyGradient(lightingSkyHash(uint2(q0.x, q1.y) + s), f - float2(0.0f, 1.0f));
    const float d = lightingSkyGradient(lightingSkyHash(uint2(q1.x, q1.y) + s), f - float2(1.0f, 1.0f));
    return 0.5f + 0.75f * lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// Fractal sum of at least one octave of noise. The octaves whose features are smaller than the footprint of the pixel
// (in cells of the first octave) fade to their mean instead of aliasing, which keeps the clouds from shimmering. Returns
// the sum in [0, 1] and the fraction of its amplitude that was faded out.
float2 lightingSkyFbm(float2 p, float footprint, uint octaves, uint seed) {
    float total = 0.0f;
    float removed = 0.0f;
    float amplitude = 0.5f;
    float amplitudeSum = 0.0f;
    for (uint i = 0; i < octaves; i++) {
        const float weight = 1.0f - smoothstep(0.25f, 0.5f, footprint);
        if (weight > 0.0f) {
            total += amplitude * (lightingSkyNoise(p, seed + i) - 0.5f) * weight;
        }

        removed += amplitude * (1.0f - weight);
        amplitudeSum += amplitude;

        // Rotation and scale by sqrt(5) with an integer matrix, so every octave keeps the period of the noise.
        p = float2(2.0f * p.x - p.y + 0.37f, p.x + 2.0f * p.y + 0.71f);
        footprint *= 2.2360679f;
        amplitude *= 0.5f;
    }

    return float2(0.5f + total / amplitudeSum, removed / amplitudeSum);
}

// Henyey-Greenstein phase function multiplied by 4 pi, so it's 1 for an isotropic medium.
float lightingSkyPhase(float cosine, float g) {
    const float g2 = g * g;
    return (1.0f - g2) / pow(max(1.0f + g2 - 2.0f * g * cosine, 1e-6f), 1.5f);
}

struct LightingSkyCloudLayer {
    // Position of the camera under the layer in cloud heights, cells of noise per cloud height and radius of
    // curvature of the layer in cloud heights.
    float2 origin;
    float frequency;
    float curvature;

    // xy offset of the shapes and zw offset of the warp, in cells.
    float4 offset;

    float coverage;
    float opacity;
    float warp;
    float absorption;
    float sunStrength;
    float ambientStrength;
    float belly;
    float haze;

    // Sunlight, light bounced by the ground, light of the sky from above and color of the haze at the horizon.
    float3 sunColor;
    float3 groundColor;
    float3 ambientTop;
    float3 ambientHorizon;

    uint octaves;
    uint warpOctaves;
    uint lightTaps;
};

// Layer of clouds one cloud height above the camera seen from below, with the direction and the sun in a space with y
// pointing up. Returns the color and the opacity (not premultiplied).
float4 lightingSkyClouds(LightingSkyCloudLayer layer, float3 direction, float3 sunDirection, float pixelAngle) {
    const float y = direction.y;
    if ((y <= 0.0f) || (layer.opacity <= 0.0f)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // The layer is curved like the surface of a planet so it stays at a finite distance at the horizon, where a flat
    // layer would squash the clouds into thin streaks.
    const float radius = layer.curvature;
    const float distance = sqrt((radius + 1.0f) * (radius + 1.0f) - radius * radius * (1.0f - y * y)) - radius * y;
    const float grazing = (distance + radius * y) / (radius + 1.0f);
    const float2 position = (layer.origin + direction.xz * distance) * layer.frequency;

    // Size of the pixel on the layer in cells, from its angular size and how slanted the layer is.
    const float footprint = pixelAngle * distance / max(grazing, 0.05f) * layer.frequency;

    // A low frequency noise warps the shapes. It moves slower than them, so the clouds change shape as they move.
    float2 shapePosition = position + layer.offset.xy;
    if ((layer.warpOctaves > 0) && (layer.warp > 0.0f)) {
        const float WarpFrequency = 0.25f;
        const float2 warpPosition = position * WarpFrequency + layer.offset.zw;
        const float warpX = lightingSkyFbm(warpPosition, footprint * WarpFrequency, layer.warpOctaves, 101).x;
        const float warpY = lightingSkyFbm(warpPosition + float2(17.3f, 9.1f), footprint * WarpFrequency, layer.warpOctaves, 117).x;
        shapePosition += (float2(warpX, warpY) - 0.5f) * layer.warp;
    }

    const float2 shape = lightingSkyFbm(shapePosition, footprint, max(layer.octaves, 1u), 7);

    // The detail lost to the filtering widens the edges, so distant clouds keep about the same coverage.
    const float threshold = 0.60f - 0.28f * layer.coverage;
    const float softness = shape.y * 0.5f;
    const float edge = smoothstep(threshold - softness, threshold + 0.04f + softness, shape.x);
    if (edge <= 0.0f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const float thickness = saturate((shape.x - threshold) / (0.22f + softness));

    // Clouds found towards the sun over the layer shadow this point. The taps get further apart and weigh less.
    const float taps = float(max(layer.lightTaps, 1u));
    const float2 sunStep = sunDirection.xz / max(sunDirection.y, 0.25f) * layer.frequency;
    const uint lightOctaves = max(layer.octaves, 3u) - 2u;
    const float maxDistance = 0.3f + 0.4f * (taps - 1.0f) / 3.0f;
    const float weightSum = taps * (taps + 1.0f) * 0.5f;
    float shadowDepth = 0.0f;
    for (uint i = 0; i < max(layer.lightTaps, 1u); i++) {
        const float tapDistance = maxDistance * pow((float(i) + 1.0f) / taps, 1.5f);
        const float2 tap = lightingSkyFbm(shapePosition + sunStep * tapDistance, footprint, lightOctaves, 7);
        shadowDepth += saturate((tap.x - threshold) / (0.22f + tap.y * 0.6f)) * ((taps - float(i)) / weightSum);
    }

    const float MinLight = 0.4f;
    const float lit = exp(-(shadowDepth * 1.5f + thickness * 0.8f) * layer.absorption);
    const float cosine = dot(direction, sunDirection);
    const float forward = lightingSkyPhase(cosine, 0.75f);
    const float backward = lightingSkyPhase(cosine, -0.25f);
    const float phase = min(0.35f * forward + 0.65f * backward, 2.5f);

    // Looking up shows the flat bottom of the clouds, which is darker where they're thick.
    const float belly = 1.0f - layer.belly * thickness * smoothstep(0.15f, 0.8f, y);
    float3 color = layer.sunColor * ((MinLight + (1.0f - MinLight) * lit) * phase * belly);

    // Thin edges glow when the sun is behind them.
    const float thin = 1.0f - thickness;
    color += layer.sunColor * (thin * thin * edge * min(forward, 30.0f) * 0.06f);
    color *= layer.sunStrength;
    color += (layer.ambientTop * (1.0f - 0.4f * thickness) + layer.groundColor * (0.35f * (0.5f + 0.5f * thickness))) * layer.ambientStrength;
    float alpha = edge * (1.0f - exp(-(0.4f + 3.0f * thickness) * layer.opacity));

    // Distant clouds fade into the haze of the horizon.
    const float haze = 1.0f - exp(-(distance - 1.0f) * layer.haze);
    color = lerp(color, layer.ambientHorizon, haze);
    alpha *= smoothstep(0.0f, 0.06f, y);
    return float4(color, alpha);
}

// Brightness of the sun disc (with a soft edge one pixel wide and darker limbs) and the glow around it.
float lightingSkySun(float3 direction, float3 sunDirection, float radius, float pixelAngle, float discStrength, float glowStrength) {
    const float cosine = dot(direction, sunDirection);
    if (cosine <= 0.0f) {
        return 0.0f;
    }

    // The angle from the sine is precise for the small angles of the disc, unlike the one from the cosine.
    const float angle = asin(min(length(cross(direction, sunDirection)), 1.0f));
    const float disc = saturate((radius - angle) / max(pixelAngle, 1e-5f) + 0.5f);
    const float limb = sqrt(saturate(1.0f - (angle * angle) / max(radius * radius, 1e-10f)));
    const float glow = pow(cosine, 2000.0f) * 0.8f + pow(cosine, 80.0f) * 0.12f;
    return disc * (0.65f + 0.35f * limb) * discStrength + glow * glowStrength;
}

// How likely it is that a pixel of the game's background (as displayed) is sky: blue, or bright and neutral like the
// painted clouds. Painted scenery like mountains, buildings or trees is kept, and the caller also keeps what's below the
// horizon (like a painted sea). Dark backgrounds (a night sky, a dark blue fog) are kept too, as the procedural sky is
// a daytime one.
float lightingSkyKey(float3 color, float whiteStrength, float minLuma) {
    const float maxChannel = max(color.r, max(color.g, color.b));
    const float minChannel = min(color.r, min(color.g, color.b));
    const float blue = smoothstep(0.04f, 0.16f, color.b - max(color.r, color.g * 0.85f));
    const float white = smoothstep(0.62f, 0.8f, minChannel) * (1.0f - smoothstep(0.12f, 0.25f, maxChannel - minChannel));
    const float luma = dot(color, float3(0.299f, 0.587f, 0.114f));
    return saturate(max(blue, white * whiteStrength)) * smoothstep(minLuma * 0.5f, minLuma, luma);
}

// Keeps the colors below the knee and compresses the ones above it smoothly towards 1.
float3 lightingSkyCompressHighlights(float3 color, float knee) {
    const float range = 1.0f - knee;
    return min(color, knee) + range * (1.0f - exp(-max(color - knee, 0.0f) / range));
}

// Interleaved gradient noise in [-0.5, 0.5].
float lightingSkyDither(float2 pixel) {
    return frac(52.9829189f * frac(dot(pixel, float2(0.06711056f, 0.00583715f)))) - 0.5f;
}
