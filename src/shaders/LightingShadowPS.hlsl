//
// RT64
//
// Discards the pixels of the shadow casters that the RDP wouldn't draw, so cutouts like leaves cast shaped shadows.
// The MERGED variant draws many draw calls at once: it finds the draw call of each triangle in a buffer (low 24 bits
// render index, high 8 bits LIGHTING_GBUFFER_* flags) indexed by the first triangle of the draw plus the primitive.
//

#define DYNAMIC_RENDER_PARAMS

#include "shared/rt64_lighting_params.h"

#include "LightingAlpha.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingShadowCB> gConstants : register(b0, space0);

#ifdef MERGED
StructuredBuffer<uint> gTriangleDraws : register(t0, space4);
#endif

void PSMain(in float4 vertexPosition : SV_POSITION, in float2 vertexUV : TEXCOORD, in float4 vertexColor : COLOR0
#ifdef MERGED
    , in uint primitiveIndex : SV_PrimitiveID
#endif
    )
{
#ifdef MERGED
    // Every pixel of a quad belongs to the same triangle, so this branch doesn't break the derivatives.
    const uint triangleDraw = gTriangleDraws[gConstants.renderIndex + primitiveIndex];
    if (((triangleDraw >> 24) & LIGHTING_GBUFFER_ALPHA_TESTED) == 0) {
        return;
    }

    const uint renderIndex = triangleDraw & 0xFFFFFFU;
#else
    const uint renderIndex = gConstants.renderIndex;
#endif
    if (lightingPixelDiscarded(renderIndex, vertexUV, vertexColor, 0.5f)) {
        discard;
    }
}
