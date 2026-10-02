//
// RT64
//
// Discards the pixels of the shadow casters that the RDP wouldn't draw, so cutouts like leaves cast shaped shadows.
//

#define DYNAMIC_RENDER_PARAMS

#include "shared/rt64_lighting_params.h"

#include "LightingAlpha.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingShadowCB> gConstants : register(b0, space0);

void PSMain(in float4 vertexPosition : SV_POSITION, in float2 vertexUV : TEXCOORD, in float4 vertexColor : COLOR0) {
    if (lightingPixelDiscarded(gConstants.renderIndex, vertexUV, vertexColor, 0.5f)) {
        discard;
    }
}
