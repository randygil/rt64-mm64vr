//
// RT64
//
// Helpers shared by the post effects passes.
//

#pragma once

static const float3 PostEffectsLumaWeights = float3(0.2126f, 0.7152f, 0.0722f);

float postEffectsLuma(float3 color) {
    return dot(color, PostEffectsLumaWeights);
}

// Interleaved gradient noise (Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", 2014), in [0, 1).
float postEffectsNoise(float2 pixel) {
    return frac(52.9829189f * frac(dot(pixel, float2(0.06711056f, 0.00583715f))));
}

// Texture coordinates of a position in pixels, clamped to the centers of the pixels at the border of the region in use
// so the bilinear filter never reads outside of it.
float2 postEffectsClampedUV(float2 position, float2 regionSize, float2 invTextureSize) {
    return clamp(position, 0.5f, regionSize - 0.5f) * invTextureSize;
}

// Offsets in pixels of the input of the downsampling filters, relative to the corner shared by the 2x2 pixels covered
// by the output pixel. Each bilinear sample at an integer offset averages a 2x2 block.
//
// Wide filter, 13 samples over 6x6 pixels (Jimenez 2014):   Narrow filter, 5 samples over 4x4 pixels (Bjorge,
//   0 . 1 . 2                                                "Bandwidth-Efficient Rendering", SIGGRAPH 2015):
//   . 3 . 4 .                                                  1 . 2
//   5 . 6 . 7                                                  . 0 .
//   . 8 . 9 .                                                  3 . 4
//  10 . 11. 12
static const float2 PostEffectsWideOffsets[13] = {
    float2(-2.0f, -2.0f), float2(0.0f, -2.0f), float2(2.0f, -2.0f),
    float2(-1.0f, -1.0f), float2(1.0f, -1.0f),
    float2(-2.0f, 0.0f), float2(0.0f, 0.0f), float2(2.0f, 0.0f),
    float2(-1.0f, 1.0f), float2(1.0f, 1.0f),
    float2(-2.0f, 2.0f), float2(0.0f, 2.0f), float2(2.0f, 2.0f)
};

static const float2 PostEffectsNarrowOffsets[5] = {
    float2(0.0f, 0.0f), float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(-1.0f, 1.0f), float2(1.0f, 1.0f)
};

// Five overlapping 2x2 boxes: the center one weighs 0.5 and the four corner ones 0.125. With the Karis average each box
// is also weighted by the inverse of its luminance, so a single bright pixel can't make the whole block flicker.
float3 postEffectsCombineWide(float3 s[13], bool karisAverage) {
    const float3 boxes[5] = {
        (s[3] + s[4] + s[8] + s[9]) * 0.25f,
        (s[0] + s[1] + s[5] + s[6]) * 0.25f,
        (s[1] + s[2] + s[6] + s[7]) * 0.25f,
        (s[5] + s[6] + s[10] + s[11]) * 0.25f,
        (s[6] + s[7] + s[11] + s[12]) * 0.25f
    };

    const float BoxWeights[5] = { 0.5f, 0.125f, 0.125f, 0.125f, 0.125f };
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    [unroll]
    for (uint i = 0; i < 5; i++) {
        const float weight = BoxWeights[i] * (karisAverage ? (1.0f / (1.0f + postEffectsLuma(boxes[i]))) : 1.0f);
        sum += boxes[i] * weight;
        weightSum += weight;
    }

    return sum / weightSum;
}

float3 postEffectsCombineNarrow(float3 s[5]) {
    return s[0] * 0.5f + (s[1] + s[2] + s[3] + s[4]) * 0.125f;
}
