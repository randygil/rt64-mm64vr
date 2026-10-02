//
// RT64
//
// Stores the normal of the opaque surfaces of a projection, for the pixels where the surface is the one left in the
// depth buffer. Normals come from the vertices when the game lit them or RT64 smoothed them, and from the faces
// otherwise. Foliage drawn as flat cards (FOLIAGE variant) gets the normals of a sphere around the center of each card,
// found by intersecting the view ray with it, so all the crossed cards of a tree shade like the same volume.
//
// The MERGED variant draws many draw calls at once: it finds the draw call and flags of each triangle in a buffer (low 24
// bits render index, high 8 bits LIGHTING_GBUFFER_* flags) indexed by the first triangle of the draw plus the primitive.
//
// Output: xy octahedral normal, z radius of the foliage sphere / 4096 (0 if it's not foliage), w = 1 where something
// was stored.
//

#define DYNAMIC_RENDER_PARAMS

#ifdef MERGED
#   define FOLIAGE
#endif

#include "shared/rt64_lighting_params.h"

#include "Depth.hlsli"
#include "LightingAlpha.hlsli"
#include "LightingCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingGBufferCB> gConstants : register(b0, space0);
ByteAddressBuffer posBuffer : register(t27, space0);
ByteAddressBuffer indexBuffer : register(t35, space0);
#ifdef MERGED
StructuredBuffer<uint> gTriangleDraws : register(t0, space4);
#endif

#ifdef MULTISAMPLING
Texture2DMS<float> gBackgroundDepth : register(t2, space3);

// The nearest surface of the pixel, which is the one the composition reads the normal for.
float sceneDepth(int2 pixel) {
    uint width, height, sampleCount;
    gBackgroundDepth.GetDimensions(width, height, sampleCount);
    float depth = 1.0f;
    for (uint s = 0; s < sampleCount; s++) {
        depth = min(depth, gBackgroundDepth.Load(pixel, s));
    }

    return depth;
}
#else
Texture2D<float> gBackgroundDepth : register(t2, space3);

float sceneDepth(int2 pixel) {
    return gBackgroundDepth.Load(int3(pixel, 0));
}
#endif

float3 loadWorldPosition(uint vertexIndex) {
    return asfloat(posBuffer.Load3(vertexIndex * 16));
}

void PSMain(in float4 vertexPosition : SV_POSITION, in float2 vertexUV : TEXCOORD, in float4 vertexColor : COLOR0, in float3 worldPosition : POSITION1, in float4 worldNormal : NORMAL0
#ifdef FOLIAGE
    , in uint primitiveIndex : SV_PrimitiveID
#endif
    , out float4 resultNormal : SV_TARGET0)
{
    // Only the surface that ended up in the depth buffer is stored.
    const int2 pixel = int2(vertexPosition.xy);
    const float surfaceDepth = sceneDepth(pixel);
    const float depthSlope = abs(ddx(vertexPosition.z)) + abs(ddy(vertexPosition.z));
    const float tolerance = max(CoplanarDepthTolerance(surfaceDepth), depthSlope * 1.5f);
    if (abs(vertexPosition.z - surfaceDepth) > tolerance) {
        discard;
    }

#ifdef MERGED
    const uint triangleDraw = gTriangleDraws[gConstants.renderIndex + primitiveIndex];
    const uint renderIndex = triangleDraw & 0xFFFFFFU;
    const uint drawFlags = triangleDraw >> 24;
#else
    const uint renderIndex = gConstants.renderIndex;
    const uint drawFlags = gConstants.flags;
#endif

    if ((drawFlags & LIGHTING_GBUFFER_ALPHA_TESTED) && lightingPixelDiscarded(renderIndex, vertexUV, vertexColor, 0.125f)) {
        discard;
    }

    // Normal of the face, pointing towards the camera.
    const float3 toCamera = gConstants.cameraPosition.xyz - worldPosition;
    float3 faceNormal = cross(ddy(worldPosition), ddx(worldPosition));
    const float faceNormalLength = length(faceNormal);
    faceNormal = (faceNormalLength > 1e-12f) ? (faceNormal / faceNormalLength) : normalize(toCamera);
    const bool backFacing = (dot(faceNormal, toCamera) < 0.0f);
    if (backFacing) {
        faceNormal = -faceNormal;
    }

    float3 normal = faceNormal;
    const bool vertexNormalValid = (worldNormal.w > 1.5f) || ((drawFlags & LIGHTING_GBUFFER_RSP_LIT) && (dot(worldNormal.xyz, worldNormal.xyz) > 1e-4f));
    if (vertexNormalValid) {
        normal = normalize(worldNormal.xyz);
        if (backFacing) {
            normal = -normal;
        }

        // Keep the normal in the visible side of the face.
        const float facing = dot(normal, faceNormal);
        if (facing < 0.05f) {
            normal = normalize(normal + faceNormal * (0.05f - facing));
        }
    }

    // Relief from the brightness of the texture, treated as a height map: the normal tilts along the directions the
    // texture coordinates follow on the surface (cotangent frame from the screen space derivatives).
    if (drawFlags & LIGHTING_GBUFFER_BUMP) {
        const float2 ddxUV = ddx(vertexUV);
        const float2 ddyUV = ddy(vertexUV);
        const float3 LumaWeights = float3(0.2126f, 0.7152f, 0.0722f);
        const float heightCenter = dot(lightingSampleTexture0(renderIndex, vertexUV, ddxUV, ddyUV).rgb, LumaWeights);
        const float heightU = dot(lightingSampleTexture0(renderIndex, vertexUV + float2(1.0f, 0.0f), ddxUV, ddyUV).rgb, LumaWeights);
        const float heightV = dot(lightingSampleTexture0(renderIndex, vertexUV + float2(0.0f, 1.0f), ddxUV, ddyUV).rgb, LumaWeights);
        const float3 dp1 = ddx(worldPosition);
        const float3 dp2 = ddy(worldPosition);
        const float3 dp2Perp = cross(dp2, normal);
        const float3 dp1Perp = cross(normal, dp1);
        float3 tangentU = dp2Perp * ddxUV.x + dp1Perp * ddyUV.x;
        float3 tangentV = dp2Perp * ddxUV.y + dp1Perp * ddyUV.y;
        const float lengthU = length(tangentU);
        const float lengthV = length(tangentV);
        if ((lengthU > 1e-12f) && (lengthV > 1e-12f)) {
            const float2 slope = clamp(float2(heightU - heightCenter, heightV - heightCenter) * gConstants.bumpStrength, -1.0f, 1.0f);
            normal = normalize(normal - (tangentU / lengthU) * slope.x - (tangentV / lengthV) * slope.y);
        }
    }

    float flags = 0.0f;
#ifdef FOLIAGE
    if (drawFlags & LIGHTING_GBUFFER_FOLIAGE) {
        // The longest edge of each triangle is the diagonal of its card, whose middle is the center of the card.
        const uint baseIndex = (gConstants.indexStart + primitiveIndex * 3) * 4;
        const float3 p0 = loadWorldPosition(indexBuffer.Load(baseIndex + 0));
        const float3 p1 = loadWorldPosition(indexBuffer.Load(baseIndex + 4));
        const float3 p2 = loadWorldPosition(indexBuffer.Load(baseIndex + 8));
        const float l01 = dot(p1 - p0, p1 - p0);
        const float l12 = dot(p2 - p1, p2 - p1);
        const float l20 = dot(p0 - p2, p0 - p2);
        float3 center;
        float diagonalSqr;
        if ((l01 >= l12) && (l01 >= l20)) {
            center = (p0 + p1) * 0.5f;
            diagonalSqr = l01;
        }
        else if (l12 >= l20) {
            center = (p1 + p2) * 0.5f;
            diagonalSqr = l12;
        }
        else {
            center = (p2 + p0) * 0.5f;
            diagonalSqr = l20;
        }

        // Sphere impostor: the normal is the one of the sphere where the view ray through the pixel enters it, or the
        // one of its silhouette if the ray misses it.
        const float radiusSqr = diagonalSqr * 0.25f;
        const float3 rayOrigin = gConstants.cameraPosition.xyz;
        const float3 rayDirection = normalize(worldPosition - rayOrigin);
        const float3 originToCenter = center - rayOrigin;
        const float closestDistance = dot(originToCenter, rayDirection);
        const float3 closestOffset = rayDirection * closestDistance - originToCenter;
        const float missSqr = dot(closestOffset, closestOffset);
        if (missSqr < radiusSqr) {
            const float3 entry = rayOrigin + rayDirection * (closestDistance - sqrt(radiusSqr - missSqr));
            normal = normalize(entry - center);
        }
        else {
            normal = normalize(closestOffset);
        }

        flags = clamp(sqrt(radiusSqr) / 4096.0f, 1.0f / 4096.0f, 1.0f);
    }
#endif

    resultNormal = float4(lightingEncodeNormal(normal), flags, 1.0f);
}
