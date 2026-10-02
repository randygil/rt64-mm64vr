//
// RT64
//
// Final pass of the post effects, drawn over the region of the scene in the color target: sharpens the scene, adds the
// bloom and the light shafts, grades the color and dithers the result. The pipeline only writes the color channels, as
// the alpha channel holds the coverage of the RDP.
//

#include "PostEffectsCommon.hlsli"
#include "PostEffectsParams.hlsli"

[[vk::push_constant]] ConstantBuffer<PostEffectsComposeCB> gConstants : register(b0, space0);
Texture2D<float4> gSceneColor : register(t1, space0);
Texture2D<float4> gBloom : register(t2, space0);
Texture2D<float4> gShafts : register(t3, space0);
SamplerState gLinearSampler : register(s4, space0);

float3 loadScene(int2 pixel) {
    return gSceneColor.Load(int3(clamp(pixel, gConstants.rect.xy, gConstants.rect.zw), 0)).rgb;
}

// Contrast adaptive sharpening, ported from CasFilter (without scaling) in ffx_cas.h of AMD FidelityFX CAS:
//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
float3 sharpen(int2 pixel, float3 e) {
    //  a b c
    //  d e f
    //  g h i
    const float3 b = loadScene(pixel + int2(0, -1));
    const float3 d = loadScene(pixel + int2(-1, 0));
    const float3 f = loadScene(pixel + int2(1, 0));
    const float3 h = loadScene(pixel + int2(0, 1));
    float3 mn = min(min(min(d, e), min(f, b)), h);
    float3 mx = max(max(max(d, e), max(f, b)), h);
    float limit = 1.0f;
    if (gConstants.flags & POST_EFFECTS_FLAG_SHARPEN_DIAGONALS) {
        // Soft minimum and maximum of the cross and the whole 3x3 neighborhood, scaled by two.
        const float3 a = loadScene(pixel + int2(-1, -1));
        const float3 c = loadScene(pixel + int2(1, -1));
        const float3 g = loadScene(pixel + int2(-1, 1));
        const float3 i = loadScene(pixel + int2(1, 1));
        mn += min(min(mn, a), min(min(c, g), i));
        mx += max(max(mx, a), max(max(c, g), i));
        limit = 2.0f;
    }

    // Less sharpening where the neighborhood is close to black or white, so it doesn't clip or ring.
    const float3 amount = sqrt(saturate(min(mn, limit - mx) / max(mx, 1e-5f)));

    //  0 w 0
    //  w 1 w
    //  0 w 0
    const float w = amount.g * gConstants.sharpenPeak;
    return saturate(((b + d + f + h) * w + e) / (1.0f + 4.0f * w));
}

float3 sampleTexture(Texture2D<float4> tex, float2 position, float2 regionSize, float2 invTextureSize) {
    return tex.SampleLevel(gLinearSampler, postEffectsClampedUV(position, regionSize, invTextureSize), 0).rgb;
}

float3 grade(float3 color, float2 scenePosition) {
    // White balance towards warm (positive) or cool (negative) light, keeping the luminance.
    if (gConstants.temperature != 0.0f) {
        const float luma = postEffectsLuma(color);
        color *= float3(1.0f + 0.1f * gConstants.temperature, 1.0f, 1.0f - 0.1f * gConstants.temperature);
        color = saturate(color * (luma / max(postEffectsLuma(color), 1e-5f)));
    }

    // Contrast as a blend towards a smooth S-curve, which keeps black and white where they are.
    color = lerp(color, color * color * (3.0f - 2.0f * color), gConstants.contrast);

    // Vibrance (the idea of SweetFX's Vibrance, MIT license) saturates the least saturated colors the most, so skin and
    // pale surfaces gain color before the already vivid ones clip.
    const float luma = postEffectsLuma(color);
    const float colorSaturation = max(color.r, max(color.g, color.b)) - min(color.r, min(color.g, color.b));
    const float vibrance = gConstants.vibrance * (1.0f - sign(gConstants.vibrance) * colorSaturation);
    color = lerp(luma.xxx, color, (1.0f + gConstants.saturation) * (1.0f + vibrance));

    // Vignette that only darkens the corners.
    const float2 rectSize = float2(gConstants.rect.zw - gConstants.rect.xy + 1);
    const float2 centered = (scenePosition / rectSize) * 2.0f - 1.0f;
    color *= 1.0f - gConstants.vignette * smoothstep(0.4f, 2.0f, dot(centered, centered));
    return color;
}

float4 PSMain(in float4 pixelPosition : SV_POSITION) : SV_TARGET {
    const int2 pixel = int2(pixelPosition.xy);
    float3 color = loadScene(pixel);
    if (gConstants.sharpenPeak < 0.0f) {
        color = sharpen(pixel, color);
    }

    // Bloom and light shafts are added with a screen blend, which brightens the dark parts the most and never clips.
    const float2 scenePosition = pixelPosition.xy - float2(gConstants.rect.xy);
    float3 light = float3(0.0f, 0.0f, 0.0f);
    if (gConstants.bloomStrength > 0.0f) {
        light += sampleTexture(gBloom, scenePosition * 0.5f, gConstants.bloomSize, gConstants.bloomInvSize) * gConstants.bloomStrength;
    }

    if (any(gConstants.shaftsColor > 0.0f)) {
        light += sampleTexture(gShafts, scenePosition * gConstants.shaftsScale, gConstants.shaftsSize, gConstants.shaftsInvSize).r * gConstants.shaftsColor;
    }

    color = 1.0f - (1.0f - color) * (1.0f - saturate(light));
    color = grade(color, scenePosition);

    // Triangular noise of up to one step of the color target hides the banding of the smooth gradients added above.
    const float noise = postEffectsNoise(pixelPosition.xy) + postEffectsNoise(pixelPosition.xy + float2(57.0f, 19.0f)) - 1.0f;
    color += noise * gConstants.ditherAmplitude;
    return float4(saturate(color), 1.0f);
}
