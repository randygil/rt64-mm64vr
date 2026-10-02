//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct FrameParams {
        uint frameCount;
        uint viewUbershaders;
        float ditherNoiseStrength;

        // Cutouts write partial coverage at their edges when multisampling instead of being cut per pixel.
        uint cutoutAntialiasing;
    };
#ifdef HLSL_CPU
};
#endif