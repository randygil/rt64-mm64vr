//
// RT64
//
// Parameters of the post effects passes, shared by their shaders and render/rt64_post_effects.cpp. Positions are in
// pixels with the centers of the pixels at .5. Every pass reads its region of the textures through clamped coordinates,
// as the textures can be bigger than the region in use.
//

#pragma once

#include "shared/rt64_hlsl.h"

// The prefilter and the downsampling use the wide 13 tap filter instead of the 5 tap one, and the upsampling a 3x3 tent
// instead of 4 bilinear samples.
#define POST_EFFECTS_FLAG_WIDE_FILTER       0x1

// The prefilter computes the mask the light shafts are blurred from.
#define POST_EFFECTS_FLAG_SHAFT_MASK        0x2

// The light shafts offset their samples with noise, which hides the banding when they're done in a single pass.
#define POST_EFFECTS_FLAG_JITTER            0x4

// The sharpening also looks at the diagonal neighbors.
#define POST_EFFECTS_FLAG_SHARPEN_DIAGONALS 0x8

#ifdef HLSL_CPU
namespace interop {
#endif
    struct PostEffectsPrefilterCB {
        // Pixels of the scene in the color copy (left, top, right, bottom; all inclusive).
        int4 sourceRect;

        // Position of the sun in pixels of the color copy and the inverse of the radius of its glow in the shaft mask.
        float2 sunPosition;
        float sunInvRadius;

        // Depth from which a pixel is background (nothing was drawn on it, so it shows the sky).
        float backgroundDepth;

        // Soft threshold on the luminance of the bloom.
        float bloomThreshold;
        float bloomKnee;

        // Luminance of the sky from which it emits light shafts.
        float shaftThreshold;
        uint flags;
    };

    struct PostEffectsDownsampleCB {
        // Inverse size of the input texture and size of its region in use.
        float2 inputInvSize;
        float2 inputSize;
        uint flags;
        uint3 padding;
    };

    struct PostEffectsUpsampleCB {
        float2 inputInvSize;
        float2 inputSize;

        // Radius of the tent filter in pixels of the input.
        float radius;

        // Blend of the upsampled level over the current one.
        float scatter;
        uint flags;
        uint padding;
    };

    struct PostEffectsShaftsCB {
        // Channel of the input the mask is read from.
        float4 channelMask;
        float2 inputInvSize;
        float2 inputSize;

        // Position of the sun in pixels of the input.
        float2 sunPosition;

        // Pixels of the input per pixel of the output.
        float inputScale;

        // Fraction of the distance to the sun covered by the samples.
        float density;

        // Weight of each sample relative to the previous one, from the pixel towards the sun.
        float decay;
        uint sampleCount;
        uint flags;
        uint padding;
    };

    struct PostEffectsComposeCB {
        // Pixels of the scene (left, top, right, bottom; all inclusive).
        int4 rect;

        // Bloom texture (half the resolution of the scene).
        float2 bloomInvSize;
        float2 bloomSize;

        // Light shafts texture and its pixels per pixel of the scene.
        float2 shaftsInvSize;
        float2 shaftsSize;
        float3 shaftsColor;
        float bloomStrength;
        float shaftsScale;

        // Negative weight of the neighbors in the sharpening filter, 0 if disabled.
        float sharpenPeak;

        // Color grading. Zero leaves the image as it is.
        float contrast;
        float vibrance;
        float saturation;
        float temperature;
        float vignette;

        // Amplitude of the dithering noise in the units of the color target.
        float ditherAmplitude;
        uint flags;
        uint3 padding;
    };
#ifdef HLSL_CPU
};
#endif
