//
// RT64
//
// Draws the scene from the sun for the shadow map of the enhanced lighting, using the positions of the vertices in
// world space instead of the ones on the screen.
//

#include "shared/rt64_lighting_params.h"

[[vk::push_constant]] ConstantBuffer<LightingShadowCB> gConstants : register(b0, space0);

void VSMain(in float4 iPosition : POSITION, in float2 iUV : TEXCOORD, in float4 iColor : COLOR, out float4 oPosition : SV_POSITION, out float2 oUV : TEXCOORD, out float4 oColor : COLOR0) {
    oPosition = mul(gConstants.shadowMatrix, float4(iPosition.xyz, 1.0f));
    oUV = iUV;
    oColor = iColor;
}
