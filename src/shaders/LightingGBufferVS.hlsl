//
// RT64
//
// Draws the opaque surfaces of a projection again to store their normals. The position on the screen is transformed
// exactly like the raster vertex shader does, so the pixels match the depth buffer. The world position and normal of
// each vertex are read with the index of the vertex.
//

#include "shared/rt64_lighting_params.h"

#include "FbRendererCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingGBufferCB> gConstants : register(b0, space0);
ByteAddressBuffer posBuffer : register(t27, space0);
ByteAddressBuffer normBuffer : register(t28, space0);

void VSMain(in uint vertexIndex : SV_VertexID, in float4 iPosition : POSITION, in float2 iUV : TEXCOORD, in float4 iColor : COLOR,
    out float4 oPosition : SV_POSITION, out float2 oUV : TEXCOORD, out float4 oColor : COLOR0, out float3 oWorldPosition : POSITION1, out float4 oWorldNormal : NORMAL0)
{
    float4 ndcPos = iPosition;
    ndcPos.xy -= float2(FbParams.resolution / 2);
    ndcPos.xy /= float2(FbParams.resolution.x / 2.0f, FbParams.resolution.y / -2.0f);
    ndcPos.xyz *= ndcPos.w;
    ndcPos.xy = (ndcPos.xy * gConstants.screenScale) + gConstants.screenOffset * ndcPos.w;
    oPosition = ndcPos;
    oUV = iUV;
    oColor = iColor;
    oWorldPosition = asfloat(posBuffer.Load3(vertexIndex * 16));
    oWorldNormal = asfloat(normBuffer.Load4(vertexIndex * 16));
}
