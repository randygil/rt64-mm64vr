//
// RT64
//

// Path traced renderer. Every draw call of the scene is stored as its own bottom level structure in world space and
// evaluated with the same rules as the raster ubershader (texture sampling, color combiner, alpha compare) from the
// any hit shader. The ray generation shaders then light the surfaces that were found along each ray.

#include "shared/rt64_blender.h"
#include "shared/rt64_color_combiner.h"
#include "shared/rt64_extra_params.h"
#include "shared/rt64_other_mode.h"
#include "shared/rt64_render_flags.h"
#include "shared/rt64_rsp_light.h"

#include "Color.hlsli"
#include "Constants.hlsli"
#include "FbRendererCommon.hlsli"
#include "FbRendererRT.hlsli"
#include "GlobalHitBuffers.hlsli"
#include "Math.hlsli"
#include "Random.hlsli"
#include "Ray.hlsli"
#include "TextureSampler.hlsli"
#include "Lights.hlsli"

// Surfaces whose opacity is at or above this value stop the ray from finding anything behind them.
#define OPAQUE_ALPHA_THRESHOLD 0.999f

// Cutout threshold used for surfaces that rely on coverage instead of blending for their transparency.
#define COVERAGE_CUTOUT_THRESHOLD 0.5f

// The RSP can't use more than 7 lights plus the ambient light.
#define MAX_RSP_LIGHTS 8

// Vertex data.

uint loadVertexIndex(uint index) {
    return indexBuffer.Load(index * 4);
}

float3 loadPosition(uint vertexIndex) {
    return asfloat(posBuffer.Load3(vertexIndex * 16));
}

float3 loadNormal(uint vertexIndex) {
    return asfloat(normBuffer.Load3(vertexIndex * 16));
}

// Vertices without lighting only have a normal if one was computed for them by smoothing the faces around them.
bool hasSmoothNormal(uint vertexIndex) {
    return asfloat(normBuffer.Load(vertexIndex * 16 + 12)) > 1.5f;
}

float3 loadVelocity(uint vertexIndex) {
    return asfloat(velBuffer.Load3(vertexIndex * 16));
}

float2 loadTexCoord(uint vertexIndex) {
    return asfloat(genTexCoordBuffer.Load2(vertexIndex * 8));
}

float4 loadColor(uint vertexIndex) {
    return asfloat(shadedColBuffer.Load4(vertexIndex * 16));
}

// Light counts are stored as 8-bit values for every vertex. Vertices without lights store their color in the space
// reserved for the normal, so the normal can only be used when the vertex was lit.
uint vertexLightCount(uint vertexIndex) {
    const uint lightCountWord = srcLightCounts.Load((vertexIndex / 4) * 4);
    return min((lightCountWord >> ((vertexIndex % 4) * 8)) & 0xFF, MAX_RSP_LIGHTS);
}

// Vertices lit by the RSP have the lighting baked into their color. Since the lighting is computed by the path tracer
// instead, the color is replaced with the one the vertex would have if it was fully lit by every light.
float3 unlitVertexColor(uint vertexIndex, float3 litColor, float mixFactor) {
    if (mixFactor <= 0.0f) {
        return litColor;
    }

    // Light indices are stored as 16-bit values for every vertex.
    const uint lightCount = vertexLightCount(vertexIndex);
    if (lightCount == 0) {
        return litColor;
    }

    const uint lightIndexWord = srcLightIndices.Load((vertexIndex / 2) * 4);
    const uint lightIndex = (lightIndexWord >> ((vertexIndex % 2) * 16)) & 0xFFFF;
    float3 lightSum = float3(0.0f, 0.0f, 0.0f);
    for (uint i = 0; i < lightCount; i++) {
        lightSum += RSPLightVector[lightIndex + i].col;
    }

    return lerp(litColor, saturate(lightSum), saturate(mixFactor));
}

uint instanceIndexFromRender(uint renderIndex) {
    return instanceRenderIndices[renderIndex].instanceIndex;
}

ExtraParams extraParamsFromRender(uint renderIndex) {
    return instanceExtraParams[instanceIndexFromRender(renderIndex)];
}

// Foliage.

float foliageHash(float2 p) {
    float3 p3 = frac(p.xyx * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.x + p3.y) * p3.z);
}

float foliageNoise(float2 p) {
    const float2 i = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0f - 2.0f * f);
    return lerp(lerp(foliageHash(i), foliageHash(i + float2(1.0f, 0.0f)), u.x), lerp(foliageHash(i + float2(0.0f, 1.0f)), foliageHash(i + float2(1.0f, 1.0f)), u.x), u.y);
}

// Height of a surface covered by small rounded leaves, from the distance to the closest leaf center of a cellular
// pattern, with some noise so the leaves aren't all the same.
// Returns the height in x and its gradient in yz.
float3 foliageDetailAt(float2 uv) {
    const float2 cell = floor(uv);
    const float2 f = uv - cell;
    float closest = 8.0f;
    float2 closestOffset = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; y++) {
        [unroll]
        for (int x = -1; x <= 1; x++) {
            const float2 neighbor = float2(x, y);
            const float2 center = float2(foliageHash(cell + neighbor), foliageHash(cell + neighbor + 19.19f));
            const float2 d = neighbor + center - f;
            const float distanceSquared = dot(d, d);
            if (distanceSquared < closest) {
                closest = distanceSquared;
                closestOffset = d;
            }
        }
    }

    const float leaf = 1.0f - closest * 1.4f;
    const float2 gradient = (leaf > 0.0f) ? (closestOffset * 2.8f * 0.75f) : 0.0f;
    return float3(saturate(leaf) * 0.75f + foliageNoise(uv * 0.5f) * 0.25f, gradient);
}

// Position in the world for effects that must stay in place when the camera moves, if the game tells where it is.
float3 foliageWorldPosition(float3 position) {
    const float3 p = position - ((RtParams.worldOrigin.w > 0.0f) ? RtParams.worldOrigin.xyz : float3(0.0f, 0.0f, 0.0f));
    return float3(dot(p, RtParams.worldRight.xyz), dot(p, RtParams.worldUp.xyz), dot(p, RtParams.worldForward.xyz));
}

// Surface evaluation.

struct SurfaceHit {
    float4 color;
    float3 normal;
    float3 velocity;
    float fog;
    bool opaque;
};

bool evaluateSurface(uint renderIndex, uint triangleIndex, float2 attribBarycentrics, float3 rayDirection, float rayT, RayDiff rayDiff,
    bool applyCulling, bool computeGradients, out SurfaceHit hit)
{
    hit.color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    hit.normal = float3(0.0f, 0.0f, 0.0f);
    hit.velocity = float3(0.0f, 0.0f, 0.0f);
    hit.fog = 0.0f;
    hit.opaque = false;

    const RenderIndices renderIndices = instanceRenderIndices[renderIndex];
    const uint instanceIndex = renderIndices.instanceIndex;
    const RenderParams rp = DynamicRenderParams[instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };
    const ExtraParams extraParams = instanceExtraParams[instanceIndex];
    const uint indexStart = renderIndices.faceIndicesStart + triangleIndex * 3;
    const uint v0 = loadVertexIndex(indexStart + 0);
    const uint v1 = loadVertexIndex(indexStart + 1);
    const uint v2 = loadVertexIndex(indexStart + 2);
    const float3 p0 = loadPosition(v0);
    const float3 p1 = loadPosition(v1);
    const float3 p2 = loadPosition(v2);
    const float3 edge01 = p1 - p0;
    const float3 edge02 = p2 - p0;
    float3 faceNormal = cross(edge01, edge02);
    const float faceNormalLength = length(faceNormal);
    faceNormal = (faceNormalLength > 1e-12f) ? (faceNormal / faceNormalLength) : -rayDirection;

    // Counter-clockwise triangles are front facing. Only back faces can be culled by the RDP's culling modes.
    const bool backFacing = dot(faceNormal, rayDirection) > 0.0f;
    if (applyCulling && renderFlagCulling(rp.flags) && backFacing) {
        return false;
    }

    const float3 barycentrics = float3(1.0f - attribBarycentrics.x - attribBarycentrics.y, attribBarycentrics.x, attribBarycentrics.y);

    // Shade color.
    float4 c0 = loadColor(v0);
    float4 c1 = loadColor(v1);
    float4 c2 = loadColor(v2);
    c0.rgb = unlitVertexColor(v0, c0.rgb, extraParams.rspLightDiffuseMix);
    c1.rgb = unlitVertexColor(v1, c1.rgb, extraParams.rspLightDiffuseMix);
    c2.rgb = unlitVertexColor(v2, c2.rgb, extraParams.rspLightDiffuseMix);
    float4 shadeColor = c0 * barycentrics.x + c1 * barycentrics.y + c2 * barycentrics.z;
    if (!renderFlagSmoothShade(rp.flags)) {
        shadeColor.rgb = c0.rgb;
    }

    if (RtParams.debugFlags & 0x10) {
        hit.color = float4(1.0f, 0.0f, 1.0f, 1.0f);
        hit.normal = -rayDirection;
        hit.opaque = true;
        return true;
    }

    // Texture coordinates and their differentials.
    const float2 uv0 = loadTexCoord(v0);
    const float2 uv1 = loadTexCoord(v1);
    const float2 uv2 = loadTexCoord(v2);
    const float2 vertexUV = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

    // Upright cutout sprites without lighting are usually foliage, like the trees and bushes drawn as flat cards. The
    // wind sways their texture inside the card, more at the top than at the bottom, like the leaves would move.
    const bool foliageSprite = otherMode.cvgXAlpha() && !Blender::usesAlphaBlend(otherMode) && renderFlagUsesTexture0(rp.flags) && (vertexLightCount(v0) == 0) && (abs(dot(faceNormal, RtParams.worldUp.xyz)) < 0.5f);
    float2 sampleUV = vertexUV;
    if (foliageSprite && (RtParams.foliageWind > 0.0f)) {
        const float3 hitWorldPosition = foliageWorldPosition(p0 * barycentrics.x + p1 * barycentrics.y + p2 * barycentrics.z);
        const float3 up = RtParams.worldUp.xyz;
        const float heightMin = min(dot(p0, up), min(dot(p1, up), dot(p2, up)));
        const float heightMax = max(dot(p0, up), max(dot(p1, up), dot(p2, up)));
        const float heightFactor = saturate((dot(p0 * barycentrics.x + p1 * barycentrics.y + p2 * barycentrics.z, up) - heightMin) / max(heightMax - heightMin, 1e-3f));

        // The phase depends on the position in the world, so gusts travel through the forest.
        const float phase = dot(hitWorldPosition.xz, float2(0.0013f, 0.0009f));
        const float t = RtParams.skyTime;
        const float sway = sin(t * 1.3f - phase * 6.0f) * 0.55f + sin(t * 2.9f - phase * 9.0f + 1.7f) * 0.25f + sin(t * 0.37f - phase * 2.0f) * 0.45f;
        sampleUV.x += sway * heightFactor * heightFactor * RtParams.foliageWind;
    }

    float2 dUVdx = float2(0.0f, 0.0f);
    float2 dUVdy = float2(0.0f, 0.0f);
    if (computeGradients) {
        RayDiff propRayDiff = propagateRayDiffs(rayDiff, rayDirection, rayT, faceNormal);
        float2 dBarydx, dBarydy;
        computeBarycentricDifferentials(propRayDiff, edge01, edge02, faceNormal, dBarydx, dBarydy);
        computeTextureDifferentials(dBarydx, dBarydy, uv0, uv1, uv2, dUVdx, dUVdy);
    }

    int tileIndex0 = 0;
    int tileIndex1 = 1;
    float lodFraction = 0.0f;
    computeLOD(otherMode, renderIndices.rdpTileCount, instanceRDPParams[instanceIndex].primLOD, 1.0f, dUVdx.x, dUVdy.y, tileIndex0, tileIndex1, lodFraction);

    float4 texVal0 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 texVal1 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float2 bumpGradient = float2(0.0f, 0.0f);
    float2 leafGradient = float2(0.0f, 0.0f);
    if (renderFlagUsesTexture0(rp.flags)) {
        const uint globalTileIndex = renderIndices.rdpTileIndex + tileIndex0;
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = renderCMS0(rp.flags);
            rdpTile.cmt = renderCMT0(rp.flags);
            rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
        }

        texVal0 = sampleTexture(otherMode, rp.flags, sampleUV, dUVdx, dUVdy, rdpTile, GPUTiles[globalTileIndex], false, computeGradients && (RtParams.textureSmoothing > 0.0f));

        // Leaves get finer detail than the texture has when seen up close: brightness variation and relief.
        if (computeGradients && foliageSprite && (RtParams.foliageDetail > 0.0f)) {
            const float green = saturate((texVal0.g - max(texVal0.r, texVal0.b)) * 6.0f);
            const float footprint = sqrt(max(dot(dUVdx, dUVdx), dot(dUVdy, dUVdy)));
            const float detailAmount = green * (1.0f - smoothstep(0.35f, 1.1f, footprint)) * RtParams.foliageDetail;
            if (detailAmount > 0.0f) {
                const float2 q = sampleUV * 1.7f;
                const float3 detail = foliageDetailAt(q);
                texVal0.rgb *= lerp(1.0f, lerp(0.72f, 1.2f, detail.x), detailAmount);
                leafGradient = detail.yz * 0.7f * detailAmount;
            }
        }

        // Relief derived from the brightness of the texture, which is treated as a height map. Only the surfaces seen
        // directly use it, the bounces don't need that much detail.
        if (computeGradients && (RtParams.bumpStrength > 0.0f)) {
            const float3 LumaWeights = float3(0.2126f, 0.7152f, 0.0722f);
            const float heightCenter = dot(texVal0.rgb, LumaWeights);
            const float heightU = dot(sampleTexture(otherMode, rp.flags, sampleUV + float2(1.0f, 0.0f), dUVdx, dUVdy, rdpTile, GPUTiles[globalTileIndex], false).rgb, LumaWeights);
            const float heightV = dot(sampleTexture(otherMode, rp.flags, sampleUV + float2(0.0f, 1.0f), dUVdx, dUVdy, rdpTile, GPUTiles[globalTileIndex], false).rgb, LumaWeights);
            bumpGradient = float2(heightU - heightCenter, heightV - heightCenter);
        }
    }

    if (renderFlagUsesTexture1(rp.flags)) {
        const bool oneCycleHardwareBug = (otherMode.cycleType() == G_CYC_1CYCLE);
        const uint globalTileIndex = renderIndices.rdpTileIndex + (oneCycleHardwareBug ? tileIndex0 : tileIndex1);
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = oneCycleHardwareBug ? renderCMS0(rp.flags) : renderCMS1(rp.flags);
            rdpTile.cmt = oneCycleHardwareBug ? renderCMT0(rp.flags) : renderCMT1(rp.flags);
            rdpTile.nativeSampler = oneCycleHardwareBug ? renderFlagNativeSampler0(rp.flags) : renderFlagNativeSampler1(rp.flags);
        }

        texVal1 = sampleTexture(otherMode, rp.flags, sampleUV, dUVdx, dUVdy, rdpTile, GPUTiles[globalTileIndex], oneCycleHardwareBug, computeGradients && (RtParams.textureSmoothing > 0.0f));
    }

    // Color combiner.
    uint randomSeed = initRand(FrParams.frameCount, DispatchRaysIndex().x + DispatchRaysIndex().y * DispatchRaysDimensions().x, 16);
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };
    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texVal0;
    ccInputs.texVal1 = texVal1;
    ccInputs.primColor = instanceRDPParams[instanceIndex].primColor;
    ccInputs.shadeColor = shadeColor;
    ccInputs.envColor = instanceRDPParams[instanceIndex].envColor;
    ccInputs.keyCenter = instanceRDPParams[instanceIndex].keyCenter;
    ccInputs.keyScale = instanceRDPParams[instanceIndex].keyScale;
    ccInputs.lodFraction = lodFraction;
    ccInputs.primLodFrac = instanceRDPParams[instanceIndex].primLOD.x;
    ccInputs.noise = nextRand(randomSeed);
    ccInputs.K4 = (instanceRDPParams[instanceIndex].convertK[4] / 255.0f);
    ccInputs.K5 = (instanceRDPParams[instanceIndex].convertK[5] / 255.0f);
    float4 combinerColor;
    float alphaCompareValue;
    colorCombiner.run(ccInputs, combinerColor, alphaCompareValue);

    // Alpha compare.
    if (otherMode.alphaCompare() == G_AC_DITHER) {
        if (alphaCompareValue < nextRand(randomSeed)) {
            return false;
        }
    }
    else if (otherMode.alphaCompare() == G_AC_THRESHOLD) {
        // Smoothed textures cut at half their alpha, where the filtered edge is a curve instead of following the texels.
        float alphaThreshold = instanceRDPParams[instanceIndex].blendColor.a;
        if (computeGradients && (RtParams.textureSmoothing > 0.0f) && !Blender::usesAlphaBlend(otherMode)) {
            alphaThreshold = max(alphaThreshold, 0.5f);
        }

        if (alphaCompareValue < alphaThreshold) {
            return false;
        }
    }

    // Determine the opacity of the surface.
    const bool alphaBlend = (otherMode.cycleType() != G_CYC_COPY) && Blender::usesAlphaBlend(otherMode);
    float opacity = 1.0f;
    if (alphaBlend) {
        opacity = saturate(combinerColor.a);
    }
    else if (otherMode.cvgXAlpha() && (combinerColor.a < COVERAGE_CUTOUT_THRESHOLD)) {
        return false;
    }

    opacity = saturate(opacity * extraParams.solidAlphaMultiplier);
    if (opacity <= EPSILON) {
        return false;
    }

    // Material color adjustments.
    float3 resultColor = saturate(combinerColor.rgb);

    // Games fake the shadows of characters with dark translucent blobs drawn on the floor. The path tracer already
    // computes the real shadows, so these are removed.
    if ((RtParams.blobShadowRemoval > 0.0f) && alphaBlend && (opacity < 0.98f) && (max(resultColor.r, max(resultColor.g, resultColor.b)) < 0.1f)) {
        if (abs(dot(faceNormal, RtParams.worldUp.xyz)) > 0.85f) {
            return false;
        }
    }
    resultColor = lerp(resultColor, extraParams.diffuseColorMix.rgb, max(extraParams.diffuseColorMix.a, 0.0f));

    // Standard fog uses the shade alpha computed by the RSP as the fog factor.
    if (Blender::usesStandardFogCycle(otherMode)) {
        hit.fog = saturate(shadeColor.a);
    }

    // Shading normal. Vertices without lighting don't have normals, so the face normal is used instead.
    const float3 n0 = loadNormal(v0);
    const float3 n1 = loadNormal(v1);
    const float3 n2 = loadNormal(v2);
    float3 normal = n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z;
    const float normalLength = length(normal);
    const bool litVertices = ((RtParams.debugFlags & 0x4) == 0) && (vertexLightCount(v0) > 0) && (vertexLightCount(v1) > 0) && (vertexLightCount(v2) > 0);
    const bool smoothVertices = ((RtParams.debugFlags & 0x4) == 0) && hasSmoothNormal(v0) && hasSmoothNormal(v1) && hasSmoothNormal(v2);
    const bool validNormals = (litVertices || smoothVertices) && (dot(n0, n0) > 0.25f) && (dot(n1, n1) > 0.25f) && (dot(n2, n2) > 0.25f) && (normalLength > 1e-6f);
    normal = validNormals ? (normal / normalLength) : faceNormal;
    if (backFacing) {
        normal = -normal;
    }

    // Flat cutout sprites without normals (trees, bushes) are shaded as if they were rounded, so the sun lights one side
    // of them and they don't look like cardboard.
    const bool cutoutSprite = !litVertices && renderFlagUsesTexture0(rp.flags) && !alphaBlend && (otherMode.cvgXAlpha() || (otherMode.alphaCompare() != G_AC_NONE));
    const bool roundSprite = cutoutSprite && (RtParams.spriteVolume > 0.0f) && (abs(dot(faceNormal, RtParams.worldUp.xyz)) < 0.5f);

    // Tilt the normal along the directions the texture coordinates follow on the triangle.
    if (any(bumpGradient != 0.0f) || any(leafGradient != 0.0f) || roundSprite) {
        const float2 duv1 = uv1 - uv0;
        const float2 duv2 = uv2 - uv0;
        const float uvDeterminant = duv1.x * duv2.y - duv1.y * duv2.x;
        if (abs(uvDeterminant) > 1e-8f) {
            float3 tangentU = (edge01 * duv2.y - edge02 * duv1.y) / uvDeterminant;
            float3 tangentV = (edge02 * duv1.x - edge01 * duv2.x) / uvDeterminant;
            tangentU = tangentU - normal * dot(tangentU, normal);
            tangentV = tangentV - normal * dot(tangentV, normal);
            const float lengthU = length(tangentU);
            const float lengthV = length(tangentV);
            if ((lengthU > 1e-8f) && (lengthV > 1e-8f)) {
                tangentU /= lengthU;
                tangentV /= lengthV;
                if (roundSprite) {
                    // The texture coordinates of a sprite's triangle span the whole sprite, so they tell where the
                    // point is between its edges. The normal leans outwards towards the edges like on a sphere.
                    const float2 uvMin = min(uv0, min(uv1, uv2));
                    const float2 uvMax = max(uv0, max(uv1, uv2));
                    const float2 local = clamp((vertexUV - (uvMin + uvMax) * 0.5f) / max((uvMax - uvMin) * 0.5f, 1e-6f), -1.0f, 1.0f);
                    normal = normalize(normal + (tangentU * local.x + tangentV * local.y) * RtParams.spriteVolume);
                }

                const float2 slope = clamp(bumpGradient * RtParams.bumpStrength + leafGradient, -1.0f, 1.0f);
                normal = normalize(normal - tangentU * slope.x - tangentV * slope.y);
            }
        }
    }

    // Make sure the shading normal is never facing away from the ray.
    if (dot(normal, rayDirection) > 0.0f) {
        normal = normalize(normal - rayDirection * dot(normal, rayDirection) * 1.01f);
    }

    const float3 vel0 = loadVelocity(v0);
    const float3 vel1 = loadVelocity(v1);
    const float3 vel2 = loadVelocity(v2);
    hit.velocity = vel0 * barycentrics.x + vel1 * barycentrics.y + vel2 * barycentrics.z;
    hit.color = float4(resultColor, opacity);
    hit.normal = normal;
    hit.opaque = (opacity >= OPAQUE_ALPHA_THRESHOLD) && (extraParams.refractionFactor <= EPSILON);
    return true;
}

// Hit shaders.

[shader("anyhit")]
void SurfaceAnyHit(inout HitInfo payload, in BuiltInTriangleIntersectionAttributes attrib) {
    if (RtParams.debugFlags & 0x8) {
        return;
    }

    const uint renderIndex = InstanceID();
    SurfaceHit hit;
    if (!evaluateSurface(renderIndex, PrimitiveIndex(), attrib.barycentrics, WorldRayDirection(), RayTCurrent(), payload.rayDiff, true, true, hit)) {
        IgnoreHit();
        return;
    }

    const ExtraParams extraParams = extraParamsFromRender(renderIndex);
    const uint2 pixelIdx = DispatchRaysIndex().xy;
    const uint2 pixelDims = DispatchRaysDimensions().xy;
    const uint hitStride = pixelDims.x * pixelDims.y;
    const uint minHi = getHitBufferIndex(0, pixelIdx, pixelDims);

    // Surfaces with a depth order bias are sorted as if they were slightly closer. The bias is removed when
    // reconstructing the position from the stored distance.
    const float tval = RayTCurrent() * (1.0f - saturate(extraParams.depthOrderBias));
    uint hitCount = min(payload.nhits, MAX_HIT_QUERIES);
    if (hitCount == MAX_HIT_QUERIES) {
        // The list is full. The surface can only replace the furthest hit if it's closer than it.
        const uint lastHi = getHitBufferIndex(MAX_HIT_QUERIES - 1, pixelIdx, pixelDims);
        if (tval >= gHitVelocityDistance[lastHi].w) {
            IgnoreHit();
            return;
        }

        hitCount = MAX_HIT_QUERIES - 1;
    }

    uint hi = getHitBufferIndex(hitCount, pixelIdx, pixelDims);
    while ((hi > minHi) && (tval < gHitVelocityDistance[hi - hitStride].w)) {
        const uint lo = hi - hitStride;
        gHitVelocityDistance[hi] = gHitVelocityDistance[lo];
        gHitColor[hi] = gHitColor[lo];
        gHitNormalFog[hi] = gHitNormalFog[lo];
        gHitInstanceId[hi] = gHitInstanceId[lo];
        hi = lo;
    }

    gHitVelocityDistance[hi] = float4(hit.velocity, tval);
    gHitColor[hi] = hit.color;
    gHitNormalFog[hi] = float4(hit.normal, hit.fog);
    gHitInstanceId[hi] = renderIndex;
    payload.nhits = min(payload.nhits + 1, MAX_HIT_QUERIES);

    // Accepting an opaque hit shortens the ray so nothing behind it needs to be evaluated anymore.
    if (!hit.opaque || !(payload.flags & HIT_INFO_FLAG_ACCEPT_OPAQUE)) {
        IgnoreHit();
    }
}

[shader("closesthit")]
void SurfaceClosestHit(inout HitInfo payload, in BuiltInTriangleIntersectionAttributes attrib) {
    // No-op.
}

[shader("anyhit")]
void ShadowAnyHit(inout ShadowHitInfo payload, in BuiltInTriangleIntersectionAttributes attrib) {
    const uint renderIndex = InstanceID();
    SurfaceHit hit;
    if (!evaluateSurface(renderIndex, PrimitiveIndex(), attrib.barycentrics, WorldRayDirection(), RayTCurrent(), payload.rayDiff, false, false, hit)) {
        IgnoreHit();
        return;
    }

    const ExtraParams extraParams = extraParamsFromRender(renderIndex);
    const float alpha = saturate(hit.color.a * extraParams.shadowAlphaMultiplier);
    payload.shadowHit = max(payload.shadowHit - alpha, 0.0f);
    if (payload.shadowHit > 0.0f) {
        IgnoreHit();
    }
    else {
        AcceptHitAndEndSearch();
    }
}

[shader("closesthit")]
void ShadowClosestHit(inout ShadowHitInfo payload, in BuiltInTriangleIntersectionAttributes attrib) {
    // No-op.
}

[shader("miss")]
void SurfaceMiss(inout HitInfo payload) {
    // No-op.
}

[shader("miss")]
void ShadowMiss(inout ShadowHitInfo payload) {
    // No-op.
}

// Ray generation helpers.

float2 worldToScreenUV(float4x4 viewProj, float3 worldPos) {
    float4 clipSpace = mul(viewProj, float4(worldPos, 1.0f));
    float2 ndc = clipSpace.xy / clipSpace.w;
    return float2(0.5f + ndc.x * 0.5f, 0.5f - ndc.y * 0.5f);
}

float3 cameraRayDirection(float2 pixelPos, float2 launchDims) {
    float2 d = (pixelPos / launchDims) * 2.0f - 1.0f;
    float4 target = mul(RtParams.projectionI, float4(d.x, -d.y, 1.0f, 1.0f));
    return normalize(mul(RtParams.viewI, float4(target.xyz / target.w, 0.0f)).xyz);
}

float3 cameraRayOrigin() {
    return mul(RtParams.viewI, float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz;
}

int2 viewportPixel(float2 uv) {
    int2 pixel = int2(RtParams.viewport.xy + uv * RtParams.viewport.zw);
    uint2 backgroundDims;
    gBackgroundColor.GetDimensions(backgroundDims.x, backgroundDims.y);
    return clamp(pixel, int2(0, 0), int2(backgroundDims) - 1);
}

// Background color of the framebuffer before the scene was drawn, in linear space.
float3 sampleBackground(float2 uv) {
    return SrgbToLinear(gBackgroundColor.Load(int3(viewportPixel(uv), 0)).rgb);
}

// Uses the background of the framebuffer as an approximation of the environment in the given direction.
float3 sampleBackgroundEnvironment(float3 direction) {
    float4 clipSpace = mul(RtParams.viewProj, float4(direction, 0.0f));
    float2 ndc;
    if (clipSpace.w > EPSILON) {
        ndc = clamp(clipSpace.xy / clipSpace.w, -1.0f, 1.0f);
    }
    else {
        // Directions behind the camera use the top of the screen, which is usually the sky.
        ndc = float2(0.0f, 1.0f);
    }

    return sampleBackground(float2(0.5f + ndc.x * 0.5f, 0.5f - ndc.y * 0.5f));
}

float3 ambientLight() {
    return RtParams.ambientBaseColor.rgb + RtParams.ambientNoGIColor.rgb;
}

float FresnelReflectAmount(float3 normal, float3 incident, float reflectivity, float fresnelMultiplier) {
    float ret = pow(clamp(1.0f + dot(normal, incident), EPSILON, 1.0f), 5.0f);
    return reflectivity + ((1.0f - reflectivity) * ret * fresnelMultiplier);
}

float4 computeFog(uint renderIndex, float fogFactor) {
    const float4 fogColor = instanceRDPParams[instanceIndexFromRender(renderIndex)].fogColor;
    return float4(SrgbToLinear(fogColor.rgb), fogFactor);
}

float3 hitPosition(float3 rayOrigin, float3 rayDirection, float storedDistance, ExtraParams extraParams) {
    const float t = storedDistance / max(1.0f - saturate(extraParams.depthOrderBias), EPSILON);
    return rayOrigin + rayDirection * t;
}

float3 directLightAt(uint2 launchIndex, float3 rayDirection, ExtraParams extraParams, float3 position, float3 normal, uint maxLights, bool checkShadows) {
    return ComputeLightsRandom(SceneLights, RtParams.lightsCount, extraParams, launchIndex, rayDirection, position, normal,
        extraParams.specularColor, maxLights, checkShadows, RtParams.diSamples, FrParams.frameCount, gBlueNoise);
}

void traceSurfaces(float3 rayOrigin, float3 rayDirection, RayDiff rayDiff, bool acceptOpaque, inout HitInfo payload) {
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin = RAY_MIN_DISTANCE;
    ray.TMax = RAY_MAX_DISTANCE;
    payload.nhits = 0;
    payload.flags = acceptOpaque ? HIT_INFO_FLAG_ACCEPT_OPAQUE : 0;
    payload.rayDiff = rayDiff;
    if (RtParams.debugFlags & 0x20) {
        return;
    }

    TraceRay(SceneBVH, RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, 0, 0, ray, payload);
}

// Sun disc and the glow around it, in linear space.
float3 sunSkyGlow(float sunCosine) {
    const float SunAngularRadius = 0.012f;
    const float cosine = saturate(sunCosine);
    const float angle = acos(clamp(sunCosine, -1.0f, 1.0f));
    const float disc = 1.0f - smoothstep(SunAngularRadius * 0.8f, SunAngularRadius, angle);
    const float glow = pow(cosine, 600.0f) * 0.8f + pow(cosine, 40.0f) * 0.12f + pow(cosine, 6.0f) * 0.05f;
    return disc * RtParams.sunDiscStrength + glow;
}

// Interiors have no sun, so their brightest surfaces (screens, lamps, glowing panels) are treated as light sources.
float emissionFactor(float3 albedo) {
    if ((RtParams.emissiveStrength <= 0.0f) || (RtParams.sunDirection.w != 0.0f)) {
        return 0.0f;
    }

    const float luminance = dot(albedo, float3(0.2126f, 0.7152f, 0.0722f));
    return smoothstep(RtParams.emissiveThreshold, min(RtParams.emissiveThreshold + 0.35f, 1.0f), luminance) * RtParams.emissiveStrength;
}

float henyeyGreenstein(float cosine, float g) {
    const float g2 = g * g;
    return (1.0f - g2) / (4.0f * M_PI * pow(max(1.0f + g2 - 2.0f * g * cosine, EPSILON), 1.5f));
}

// Procedural sky. Directions are converted to a world with Y pointing up.

float3 toSkySpace(float3 direction) {
    return float3(dot(direction, RtParams.worldRight.xyz), dot(direction, RtParams.worldUp.xyz), dot(direction, RtParams.worldForward.xyz));
}

bool worldPositionKnown() {
    return RtParams.worldOrigin.w > 0.0f;
}

// Position in the game's world with Y pointing up, for games that tell where the camera is.
float3 worldPositionOf(float3 position) {
    return toSkySpace(position - RtParams.worldOrigin.xyz);
}

// Where the camera is in the units of the cloud layer. Only the horizontal position is used, so the clouds move by
// as the camera moves through the world.
float3 skyOrigin() {
    if (!worldPositionKnown()) {
        return 0.0f;
    }

    const float3 cameraPosition = worldPositionOf(float3(0.0f, 0.0f, 0.0f));
    return float3(cameraPosition.x, 0.0f, cameraPosition.z) / max(RtParams.cloudHeight, 1.0f);
}

bool proceduralSkyActive() {
    return (RtParams.skyMode > 0.0f) && (RtParams.sunDirection.w > 0.0f);
}

float skyHash(float2 p) {
    float3 p3 = frac(p.xyx * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.x + p3.y) * p3.z);
}

float skyNoise(float2 p) {
    const float2 i = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0f - 2.0f * f);
    const float a = skyHash(i);
    const float b = skyHash(i + float2(1.0f, 0.0f));
    const float c = skyHash(i + float2(0.0f, 1.0f));
    const float d = skyHash(i + float2(1.0f, 1.0f));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float skyFbm(float2 p, uint octaves) {
    float sum = 0.0f;
    float amplitude = 0.5f;
    for (uint i = 0; i < octaves; i++) {
        sum += skyNoise(p) * amplitude;
        p = float2(p.x * 1.6f - p.y * 1.2f, p.x * 1.2f + p.y * 1.6f) + float2(17.1f, 3.7f);
        amplitude *= 0.5f;
    }

    return sum;
}

float2 raySphere(float3 origin, float3 direction, float radius) {
    const float b = dot(direction, origin);
    const float c = dot(origin, origin) - radius * radius;
    const float d = b * b - c;
    if (d < 0.0f) {
        return float2(1e9f, -1e9f);
    }

    const float s = sqrt(d);
    return float2(-b - s, -b + s);
}

// Single scattering of the sun light through the atmosphere (Rayleigh and Mie), as seen from the ground.
float3 atmosphereScattering(float3 direction, float3 sunDirection) {
    const float PlanetRadius = 6371e3f;
    const float AtmosphereRadius = 6471e3f;
    const float3 RayleighCoefficient = float3(5.5e-6f, 13.0e-6f, 22.4e-6f);
    const float MieCoefficient = 21e-6f;
    const float RayleighHeight = 8e3f;
    const float MieHeight = 1.2e3f;
    const float MieAnisotropy = 0.758f;
    const uint PrimarySteps = 10;
    const uint LightSteps = 3;
    const float3 origin = float3(0.0f, PlanetRadius + 200.0f, 0.0f);
    const float pathLength = raySphere(origin, direction, AtmosphereRadius).y;
    const float stepSize = pathLength / float(PrimarySteps);
    float3 totalRayleigh = 0.0f;
    float3 totalMie = 0.0f;
    float depthRayleigh = 0.0f;
    float depthMie = 0.0f;
    for (uint i = 0; i < PrimarySteps; i++) {
        const float3 position = origin + direction * (stepSize * (float(i) + 0.5f));
        const float height = length(position) - PlanetRadius;
        const float stepRayleigh = exp(-height / RayleighHeight) * stepSize;
        const float stepMie = exp(-height / MieHeight) * stepSize;
        depthRayleigh += stepRayleigh;
        depthMie += stepMie;

        const float lightStepSize = raySphere(position, sunDirection, AtmosphereRadius).y / float(LightSteps);
        float lightRayleigh = 0.0f;
        float lightMie = 0.0f;
        for (uint j = 0; j < LightSteps; j++) {
            const float3 lightPosition = position + sunDirection * (lightStepSize * (float(j) + 0.5f));
            const float lightHeight = max(length(lightPosition) - PlanetRadius, 0.0f);
            lightRayleigh += exp(-lightHeight / RayleighHeight) * lightStepSize;
            lightMie += exp(-lightHeight / MieHeight) * lightStepSize;
        }

        const float3 attenuation = exp(-(MieCoefficient * (depthMie + lightMie) + RayleighCoefficient * (depthRayleigh + lightRayleigh)));
        totalRayleigh += stepRayleigh * attenuation;
        totalMie += stepMie * attenuation;
    }

    const float mu = dot(direction, sunDirection);
    const float g2 = MieAnisotropy * MieAnisotropy;
    const float phaseRayleigh = 3.0f / (16.0f * M_PI) * (1.0f + mu * mu);
    const float phaseMie = 3.0f / (8.0f * M_PI) * ((1.0f - g2) * (mu * mu + 1.0f)) / (pow(max(1.0f + g2 - 2.0f * mu * MieAnisotropy, EPSILON), 1.5f) * (2.0f + g2));
    return 22.0f * (phaseRayleigh * RayleighCoefficient * totalRayleigh + phaseMie * MieCoefficient * totalMie);
}

float cloudDensityAt(float2 p, uint octaves, float coverageBias) {
    // The shape is warped by a lower frequency noise so the clouds don't look like uniform blobs.
    const float2 wind = float2(1.0f, 0.35f) * RtParams.skyTime * 0.01f;
    const float2 q = p + wind;
    const float2 warp = float2(skyFbm(q * 0.35f + 5.2f, 3), skyFbm(q * 0.35f + 1.3f, 3));
    const float shape = skyFbm(q + warp * 1.2f - wind * 0.5f, octaves);
    const float threshold = lerp(0.72f, 0.3f, saturate(RtParams.cloudCoverage + coverageBias));
    return saturate((shape - threshold) * 3.5f);
}

// Layer of clouds at a fixed height, lit by the sun with self shadowing. Returns premultiplied color and coverage.
float4 skyClouds(float3 direction, float3 sunDirection, float3 skyAmbient, uint octaves) {
    if ((RtParams.cloudDensity <= 0.0f) || (direction.y < 0.01f)) {
        return 0.0f;
    }

    const float distance = 1.0f / direction.y;
    const float2 p = (skyOrigin().xz + direction.xz * distance) * 1.2f / max(RtParams.cloudScale, 0.01f);

    // There are more clouds towards the horizon, which also hides the repetition of the layer.
    const float coverageBias = (1.0f - smoothstep(0.05f, 0.45f, direction.y)) * 0.25f;
    const float density = cloudDensityAt(p, octaves, coverageBias);
    if (density <= 0.0f) {
        return 0.0f;
    }

    // Density found towards the sun darkens the parts of the cloud that face away from it.
    const float2 sunStep = sunDirection.xz / max(sunDirection.y, 0.15f) * 0.06f;
    float sunDepth = 0.0f;
    for (uint i = 1; i <= 3; i++) {
        sunDepth += cloudDensityAt(p + sunStep * float(i), min(octaves, 4u), coverageBias);
    }

    const float cosine = dot(direction, sunDirection);
    const float phase = lerp(henyeyGreenstein(cosine, -0.15f), henyeyGreenstein(cosine, 0.75f), 0.35f) * 4.0f * M_PI;
    const float sunTransmittance = exp(-sunDepth * RtParams.cloudDensity * 0.18f);
    const float powder = 1.0f - exp(-density * RtParams.cloudDensity * 0.6f);
    const float3 sunLight = RtParams.sunColor.rgb * 1.35f;
    float3 color = sunLight * lerp(0.3f, 1.0f, sunTransmittance) * lerp(1.0f, phase, 0.5f) * lerp(0.7f, 1.0f, powder);
    color += skyAmbient * lerp(1.0f, 0.75f, density);

    // Distant clouds fade into the haze of the horizon.
    float alpha = 1.0f - exp(-density * RtParams.cloudDensity);
    alpha *= smoothstep(0.01f, 0.12f, direction.y);
    return float4(color * alpha, alpha);
}


float skyHash3(float3 p) {
    p = frac(p * 0.1031f);
    p += dot(p, p.zyx + 31.32f);
    return frac((p.x + p.y) * p.z);
}

float skyNoise3(float3 p) {
    const float3 i = floor(p);
    const float3 f = frac(p);
    const float3 u = f * f * (3.0f - 2.0f * f);
    const float n000 = skyHash3(i);
    const float n100 = skyHash3(i + float3(1.0f, 0.0f, 0.0f));
    const float n010 = skyHash3(i + float3(0.0f, 1.0f, 0.0f));
    const float n110 = skyHash3(i + float3(1.0f, 1.0f, 0.0f));
    const float n001 = skyHash3(i + float3(0.0f, 0.0f, 1.0f));
    const float n101 = skyHash3(i + float3(1.0f, 0.0f, 1.0f));
    const float n011 = skyHash3(i + float3(0.0f, 1.0f, 1.0f));
    const float n111 = skyHash3(i + float3(1.0f, 1.0f, 1.0f));
    return lerp(lerp(lerp(n000, n100, u.x), lerp(n010, n110, u.x), u.y), lerp(lerp(n001, n101, u.x), lerp(n011, n111, u.x), u.y), u.z);
}

static const float CloudBottom = 1.0f;
static const float CloudTop = 1.9f;

// Where the clouds are and how tall they grow, from a 2D map that moves with the wind.
float cloudCoverageMap(float2 xz) {
    const float2 wind = float2(1.0f, 0.35f) * RtParams.skyTime * 0.012f;
    const float2 q = xz * 0.9f / max(RtParams.cloudScale, 0.01f) + wind;
    const float2 warp = float2(skyNoise(q * 0.4f + 5.2f), skyNoise(q * 0.4f + 1.3f)) * 0.75f;
    const float shape = skyFbm(q + warp * 0.9f, 4);
    const float threshold = lerp(0.7f, 0.32f, saturate(RtParams.cloudCoverage));
    return saturate((shape - threshold) * 3.0f);
}

float cloudVolumeDensity(float3 position, bool detail) {
    const float height = saturate((position.y - CloudBottom) / (CloudTop - CloudBottom));
    const float coverage = cloudCoverageMap(position.xz);

    // Cumulus shape: flat bottom and rounded tops that reach higher where the coverage is denser.
    float density = smoothstep(0.0f, 0.08f, height) * saturate((coverage * 1.1f - height) * 3.0f);
    if (detail && (density > 0.0f)) {
        // Billowy detail eats away the edges of the clouds.
        const float3 wind = float3(1.0f, 0.0f, 0.35f) * RtParams.skyTime * 0.02f;
        const float3 q = position * 7.0f + wind;
        const float erosion = skyNoise3(q) * 0.5f + skyNoise3(q * 2.3f) * 0.27f + skyNoise3(q * 5.1f) * 0.15f + skyNoise3(q * 11.3f) * 0.08f;
        density = saturate(density - (1.0f - density) * erosion * 0.9f - erosion * 0.12f);
    }

    return density;
}

// Clouds ray marched through a layer of the atmosphere. Returns premultiplied color and coverage.
float4 skyVolumetricClouds(float3 direction, float3 sunDirection, float3 skyAmbient, float jitter) {
    if ((RtParams.cloudDensity <= 0.0f) || (direction.y < 0.015f)) {
        return 0.0f;
    }

    // The layer is thinner when looking up, so it needs fewer steps there than towards the horizon.
    const float start = CloudBottom / direction.y;
    const float end = min(CloudTop / direction.y, start + 3.0f);
    const uint Steps = uint(clamp((end - start) * 9.0f, 10.0f, 22.0f));
    const float stepSize = (end - start) / float(Steps);
    const float cosine = dot(direction, sunDirection);
    const float phase = lerp(henyeyGreenstein(cosine, -0.2f), henyeyGreenstein(cosine, 0.8f), 0.3f) * 4.0f * M_PI;
    const float3 sunLight = RtParams.sunColor.rgb * float3(2.1f, 2.0f, 1.85f);
    const float3 origin = skyOrigin();
    const float extinction = RtParams.cloudDensity * 2.0f;
    float transmittance = 1.0f;
    float3 color = 0.0f;
    for (uint i = 0; i < Steps; i++) {
        const float3 position = origin + direction * (start + stepSize * (float(i) + jitter));
        const float density = cloudVolumeDensity(position, true);
        if (density <= 0.0f) {
            continue;
        }

        // Cloud between this point and the sun.
        float sunDepth = 0.0f;
        float lightStep = 0.05f;
        float3 lightPosition = position;
        for (uint j = 0; j < 3; j++) {
            lightPosition += sunDirection * lightStep;
            sunDepth += cloudVolumeDensity(lightPosition, j == 0) * lightStep;
            lightStep *= 2.0f;
        }

        const float height = saturate((position.y - CloudBottom) / (CloudTop - CloudBottom));
        const float sunTransmittance = exp(-sunDepth * extinction) * 0.7f + exp(-sunDepth * extinction * 0.25f) * 0.3f;
        const float powder = 1.0f - exp(-density * extinction * stepSize * 4.0f);
        const float3 ambient = skyAmbient * lerp(0.4f, 0.95f, height);
        const float3 scattering = sunLight * sunTransmittance * phase * lerp(0.5f, 1.0f, powder) + ambient;
        const float sampleTransmittance = exp(-density * extinction * stepSize * 8.0f);
        color += scattering * (1.0f - sampleTransmittance) * transmittance;
        transmittance *= sampleTransmittance;
        if (transmittance < 0.02f) {
            break;
        }
    }

    // Distant clouds fade into the haze of the horizon.
    const float alpha = 1.0f - transmittance;
    if (alpha > 0.001f) {
        const float haze = 1.0f - exp(-start * 0.06f);
        const float3 horizonColor = atmosphereScattering(normalize(float3(direction.x, 0.02f, direction.z)), sunDirection) * RtParams.skyExposure * RtParams.skyTint.rgb;
        color = lerp(color, horizonColor * alpha, haze);
    }

    const float fade = smoothstep(0.015f, 0.08f, direction.y);
    return float4(color * fade, alpha * fade);
}

// Amount of sun light that goes through the clouds to reach the given point.
float cloudShadowAt(float3 position) {
    if (!proceduralSkyActive() || !worldPositionKnown() || (RtParams.cloudShadows <= 0.0f) || (RtParams.cloudDensity <= 0.0f)) {
        return 1.0f;
    }

    // The ground is considered to be at the height of the camera.
    const float3 worldPosition = worldPositionOf(position);
    const float3 cameraPosition = worldPositionOf(float3(0.0f, 0.0f, 0.0f));
    const float3 sunDirection = normalize(toSkySpace(RtParams.sunDirection.xyz));
    float3 p = float3(worldPosition.x, worldPosition.y - cameraPosition.y, worldPosition.z) / max(RtParams.cloudHeight, 1.0f);
    p += sunDirection * (((CloudBottom + CloudTop) * 0.5f - p.y) / max(sunDirection.y, 0.05f));
    const float coverage = cloudCoverageMap(p.xz);
    return 1.0f - RtParams.cloudShadows * smoothstep(0.05f, 0.5f, coverage);
}

// Color of the sky in the given direction in linear space, with the sun disc if requested. Volumetric clouds are used
// if the jitter is positive.
float3 proceduralSkyColor(float3 direction, uint cloudOctaves, bool sunDisc, float jitter) {
    float3 skyDirection = toSkySpace(direction);
    skyDirection.y = max(skyDirection.y, 0.0f);
    skyDirection = normalize(skyDirection + float3(0.0f, 0.002f, 0.0f));
    const float3 sunDirection = normalize(toSkySpace(RtParams.sunDirection.xyz));
    float3 sky = atmosphereScattering(skyDirection, sunDirection) * RtParams.skyExposure;
    const float skyLuma = dot(sky, float3(0.2126f, 0.7152f, 0.0722f));
    sky = max(lerp(skyLuma.xxx, sky, RtParams.skySaturation), 0.0f) * RtParams.skyTint.rgb;

    const float3 skyAmbient = lerp(float3(0.7f, 0.76f, 0.86f), sky, 0.3f);
    float4 clouds;
    if (jitter >= 0.0f) {
        clouds = skyVolumetricClouds(skyDirection, sunDirection, skyAmbient, jitter);
    }
    else {
        clouds = skyClouds(skyDirection, sunDirection, skyAmbient, cloudOctaves);
    }

    if (sunDisc) {
        const float SunAngularRadius = 0.012f;
        const float angle = acos(clamp(dot(skyDirection, sunDirection), -1.0f, 1.0f));
        const float disc = 1.0f - smoothstep(SunAngularRadius * 0.8f, SunAngularRadius, angle);
        sky += disc * RtParams.sunDiscStrength * RtParams.sunColor.rgb;
    }

    return sky * (1.0f - clouds.a) + clouds.rgb;
}

// How likely it is that a pixel of the game's background (in sRGB) is sky: blue or bright and neutral like clouds.
float backgroundSkyKey(float3 color) {
    const float maxChannel = max(color.r, max(color.g, color.b));
    const float minChannel = min(color.r, min(color.g, color.b));
    const float blue = smoothstep(0.04f, 0.16f, color.b - max(color.r, color.g * 0.85f));
    const float cloud = smoothstep(0.62f, 0.8f, minChannel) * (1.0f - smoothstep(0.12f, 0.25f, maxChannel - minChannel));
    return saturate(max(blue, cloud));
}

// The game's background with its sky replaced by the procedural one.
float3 skyBackground(float3 direction, float3 background, float jitter) {
    const float upCosine = dot(direction, RtParams.worldUp.xyz);
    const float weight = (RtParams.skyMode >= 2.0f) ? 1.0f : backgroundSkyKey(LinearToSrgb(background));
    const float horizon = (RtParams.skyMode >= 2.0f) ? 1.0f : smoothstep(-0.02f, 0.06f, upCosine);
    if (weight * horizon <= 0.0f) {
        return background;
    }

    return lerp(background, proceduralSkyColor(direction, 7, true, jitter), weight * horizon);
}

// Approximation of the procedural sky for the bounces from colors computed once per frame, with the clouds blended
// in by how much of the sky they cover.
float3 proceduralSkyLight(float3 direction) {
    const float3 skyDirection = toSkySpace(direction);
    const float3 sunDirection = toSkySpace(RtParams.sunDirection.xyz);
    const float2 horizontal = skyDirection.xz;
    const float2 sunHorizontal = sunDirection.xz;
    const float towardsSun = (dot(horizontal, horizontal) > 1e-6f) && (dot(sunHorizontal, sunHorizontal) > 1e-6f) ? (dot(normalize(horizontal), normalize(sunHorizontal)) * 0.5f + 0.5f) : 0.5f;
    const float3 horizonColor = lerp(RtParams.skyHorizonAwayColor.rgb, RtParams.skyHorizonSunColor.rgb, towardsSun * towardsSun);
    const float3 sky = lerp(horizonColor, RtParams.skyZenithColor.rgb, sqrt(saturate(skyDirection.y)));
    const float3 cloudColor = RtParams.sunColor.rgb * 0.9f + RtParams.skyZenithColor.rgb * 0.3f;
    const float cloudCover = (RtParams.cloudDensity > 0.0f) ? saturate(RtParams.cloudCoverage) * 0.6f : 0.0f;
    return lerp(sky, cloudColor, cloudCover);
}

// The procedural sky replaces the background in the directions above the horizon by the given amount. The bounces use
// a cheaper approximation of it, while reflections need it with all its detail.
float3 sampleEnvironment(float3 direction, float skyAmount, bool detailed) {
    const float3 background = sampleBackgroundEnvironment(direction);
    if (!proceduralSkyActive() || (skyAmount <= 0.0f)) {
        return background;
    }

    const float horizon = smoothstep(-0.05f, 0.1f, dot(direction, RtParams.worldUp.xyz));
    const float3 sky = detailed ? proceduralSkyColor(direction, 4, false, -1.0f) : proceduralSkyLight(direction);
    return lerp(background, sky, horizon * skyAmount);
}

// Primary rays.

[shader("raygeneration")]
void PrimaryRayGen() {
    const uint2 launchIndex = DispatchRaysIndex().xy;
    if (RtParams.debugFlags & 0x40) {
        gDiffuse[launchIndex] = float4(1.0f, 0.0f, 1.0f, 1.0f);
        gInstanceId[launchIndex] = -1;
        return;
    }

    const uint2 launchDims = DispatchRaysDimensions().xy;
    const float2 pixelPos = launchIndex + 0.5f + RtParams.pixelJitter;
    const float2 screenUV = pixelPos / float2(launchDims);
    const float3 rayOrigin = cameraRayOrigin();
    const float3 rayDirection = cameraRayDirection(pixelPos, launchDims);

    // Ray differentials from the direction of the neighboring pixels.
    RayDiff rayDiff = zeroRayDiff();
    rayDiff.dDdx = cameraRayDirection(pixelPos + float2(1.0f, 0.0f), launchDims) - rayDirection;
    rayDiff.dDdy = cameraRayDirection(pixelPos + float2(0.0f, 1.0f), launchDims) - rayDirection;

    gViewDirection[launchIndex] = float4(rayDirection, 0.0f);
    gReflection[launchIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    gRefraction[launchIndex] = float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float3 bgColor = sampleBackground(screenUV);
    const float3 bgPosition = rayOrigin + rayDirection * RAY_MAX_DISTANCE;

    HitInfo payload;
    traceSurfaces(rayOrigin, rayDirection, rayDiff, false, payload);

    float3 resPosition = float3(0.0f, 0.0f, 0.0f);
    float3 resNormal = -rayDirection;
    float3 resSpecular = float3(0.0f, 0.0f, 0.0f);
    float3 resTransparent = float3(0.0f, 0.0f, 0.0f);
    float3 resTransparentLight = float3(0.0f, 0.0f, 0.0f);
    bool resTransparentLightComputed = false;
    float4 resColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float resLitWeight = 0.0f;
    float2 resFlow = (worldToScreenUV(RtParams.prevViewProj, bgPosition) - worldToScreenUV(RtParams.viewProj, bgPosition)) * RtParams.resolution.xy;
    float resRoughness = 1.0f;
    float resLockMask = 0.0f;
    float resDepth = 1.0f;
    int resInstanceId = -1;
    const uint hitCount = min(payload.nhits, MAX_HIT_QUERIES);
    for (uint hit = 0; hit < hitCount; hit++) {
        const uint hitBufferIndex = getHitBufferIndex(hit, launchIndex, launchDims);
        const float4 hitColor = gHitColor[hitBufferIndex];
        float alphaContrib = resColor.a * hitColor.a;
        if (alphaContrib < EPSILON) {
            continue;
        }

        const uint renderIndex = gHitInstanceId[hitBufferIndex];
        const ExtraParams extraParams = extraParamsFromRender(renderIndex);
        const float4 hitVelocityDistance = gHitVelocityDistance[hitBufferIndex];
        const float4 hitNormalFog = gHitNormalFog[hitBufferIndex];
        const float3 vertexPosition = hitPosition(rayOrigin, rayDirection, hitVelocityDistance.w, extraParams);
        const float3 vertexNormal = normalize(hitNormalFog.xyz);
        const float3 albedo = SrgbToLinear(hitColor.rgb);
        const bool usesLighting = (extraParams.lightGroupMaskBits > 0);
        const bool applyLighting = usesLighting && (hitColor.a > APPLY_LIGHTS_MINIMUM_ALPHA);
        resLockMask += extraParams.lockMask * alphaContrib;

        bool storeHit = false;
        if (hitNormalFog.w > EPSILON) {
            // Fog is blended in the same space as the RDP blender does it.
            const float4 fogColor = instanceRDPParams[instanceIndexFromRender(renderIndex)].fogColor;
            resTransparent += fogColor.rgb * hitNormalFog.w * alphaContrib;
            alphaContrib *= (1.0f - hitNormalFog.w);
        }

        // Translucent horizontal surfaces are most likely water, which reflects what's around it at grazing angles.
        float reflectionFactor = extraParams.reflectionFactor;
        float reflectionFresnelFactor = extraParams.reflectionFresnelFactor;
        if ((reflectionFactor <= EPSILON) && (RtParams.waterReflection > 0.0f) && (hitColor.a > 0.15f) && (hitColor.a < 0.95f) && (abs(dot(vertexNormal, RtParams.worldUp.xyz)) > 0.9f)) {
            reflectionFactor = RtParams.waterReflection;
            reflectionFresnelFactor = 1.0f;
        }

        if (reflectionFactor > EPSILON) {
            const float fresnelAmount = FresnelReflectAmount(vertexNormal, rayDirection, reflectionFactor, reflectionFresnelFactor);
            const float reflectAmount = fresnelAmount * alphaContrib;
            gReflection[launchIndex].a = reflectAmount;
            alphaContrib *= (1.0f - fresnelAmount);
            resLockMask += reflectAmount;
            storeHit = true;
        }

        const float3 colorAdd = albedo * alphaContrib;
        if (applyLighting) {
            resColor.rgb += colorAdd;
            resLitWeight += alphaContrib;
            storeHit = true;
        }
        else if (usesLighting) {
            // Transparent geometry that needs lighting but isn't solid enough to be the main surface of the pixel.
            // A single light sample is shared by all the surfaces of this kind.
            if (!resTransparentLightComputed) {
                resTransparentLight = directLightAt(launchIndex, rayDirection, extraParams, vertexPosition, vertexNormal, 1, true);
                resTransparentLightComputed = true;
            }

            resTransparent += LinearToSrgb(albedo * (ambientLight() + extraParams.selfLight + resTransparentLight)) * alphaContrib;
        }
        else {
            resTransparent += LinearToSrgb(albedo * (ambientLight() + extraParams.selfLight)) * alphaContrib;
        }

        resColor.a *= (1.0f - hitColor.a);

        // Refraction stops the search for more hits. The refraction rays will find whatever is behind the surface.
        if (extraParams.refractionFactor > EPSILON) {
            storeHit = true;
            gRefraction[launchIndex].a = resColor.a;
            resColor.a = 0.0f;
        }

        if (storeHit && (resInstanceId < 0)) {
            const float3 prevPosition = vertexPosition - hitVelocityDistance.xyz;
            const float4 projPos = mul(RtParams.viewProj, float4(vertexPosition, 1.0f));
            resPosition = vertexPosition;
            resNormal = vertexNormal;
            resSpecular = extraParams.specularColor;
            resRoughness = extraParams.roughnessFactor;
            resInstanceId = int(renderIndex);
            resFlow = (worldToScreenUV(RtParams.prevViewProj, prevPosition) - worldToScreenUV(RtParams.viewProj, vertexPosition)) * RtParams.resolution.xy;
            resDepth = projPos.z / projPos.w;
        }

        if (resColor.a <= EPSILON) {
            break;
        }
    }

    // The lit surfaces are stored as their average albedo and the coverage they have on the pixel. Everything else
    // (fog, unlit surfaces and the background) is blended in sRGB space like the RDP would do it.
    float3 skyColor = bgColor;
    const bool sunActive = (RtParams.sunDirection.w > 0.0f);
    const float3 sunDirection = RtParams.sunDirection.xyz;
    const float sunCosine = dot(rayDirection, sunDirection);
    if (sunActive && (resColor.a > EPSILON)) {
        if (proceduralSkyActive()) {
            skyColor = skyBackground(rayDirection, bgColor, getBlueNoise(gBlueNoise, launchIndex, FrParams.frameCount + 29).r);
        }
        else {
            skyColor += sunSkyGlow(sunCosine) * RtParams.sunColor.rgb;
        }
    }

    resTransparent += LinearToSrgb(skyColor) * resColor.a;

    // Light scattered by the air towards the camera. A single point along the view ray is tested for visibility
    // against the sun every frame and the temporal accumulation turns it into light shafts.
    if (sunActive && (RtParams.volumetricStrength > 0.0f)) {
        const float viewDistance = (resInstanceId >= 0) ? length(resPosition - rayOrigin) : RtParams.volumetricDistance;
        const float marchDistance = min(viewDistance, RtParams.volumetricDistance);
        const float noise = getBlueNoise(gBlueNoise, launchIndex, FrParams.frameCount + 17).r;
        const float3 samplePosition = rayOrigin + rayDirection * marchDistance * noise;
        const float visibility = TraceShadow(samplePosition, sunDirection, RAY_MIN_DISTANCE, RAY_MAX_DISTANCE, DEPTH_RAY_QUERY_MASK);
        const float density = marchDistance / RtParams.volumetricDistance;
        // The forward scattering peak is limited so looking towards the sun doesn't wash out the whole sky.
        const float phase = min(henyeyGreenstein(sunCosine, RtParams.volumetricAnisotropy) * 4.0f * M_PI, RtParams.volumetricMaxPhase);
        resTransparent += RtParams.sunColor.rgb * visibility * density * phase * RtParams.volumetricStrength;
    }

    resColor.rgb = (resLitWeight > EPSILON) ? (resColor.rgb / resLitWeight) : float3(0.0f, 0.0f, 0.0f);
    resColor.a = resLitWeight;
    if ((RtParams.debugFlags & 0x80) && (resInstanceId >= 0)) {
        const int3 cell = int3(floor(worldPositionOf(resPosition) / 200.0f));
        resColor.rgb = ((cell.x + cell.y + cell.z) & 1) ? float3(0.9f, 0.9f, 0.9f) : float3(0.1f, 0.1f, 0.1f);
    }

    gShadingPosition[launchIndex] = float4(resPosition, 0.0f);
    gShadingNormal[launchIndex] = float4(resNormal, 0.0f);
    gShadingSpecular[launchIndex] = float4(resSpecular, 0.0f);
    gDiffuse[launchIndex] = resColor;
    gInstanceId[launchIndex] = resInstanceId;
    gTransparent[launchIndex] = float4(resTransparent, 1.0f);
    gFlow[launchIndex] = resFlow;
    gReactiveMask[launchIndex] = min(max(resTransparent.r, max(resTransparent.g, resTransparent.b)), 0.9f);
    gLockMask[launchIndex] = RtParams.binaryLockMask ? step(0.5f, resLockMask) : min(resLockMask, 1.0f);
    gNormalRoughness[launchIndex] = float4(resNormal, resRoughness);
    gDepth[launchIndex] = resDepth;
}

// History reprojection shared by the direct and indirect light passes.

float reprojectionWeight(uint2 launchIndex, float3 normal, out int2 prevIndex) {
    const float WeightNormalExponent = 128.0f;
    const float2 flow = gFlow[launchIndex].xy;
    prevIndex = int2(float2(launchIndex) + float2(0.5f, 0.5f) + flow);
    uint2 dims = DispatchRaysDimensions().xy;
    if (any(prevIndex < 0) || any(prevIndex >= int2(dims))) {
        return 0.0f;
    }

    const float prevDepth = gPrevDepth[prevIndex];
    const float3 prevNormal = gPrevNormalRoughness[prevIndex].xyz;
    const float depth = gDepth[launchIndex];
    const float weightDepth = abs(depth - prevDepth) / 0.01f;
    const float weightNormal = pow(max(0.0f, dot(prevNormal, normal)), WeightNormalExponent);
    return exp(-weightDepth) * weightNormal;
}

// Direct light.

[shader("raygeneration")]
void DirectRayGen() {
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const int instanceId = gInstanceId[launchIndex];
    if (instanceId < 0) {
        gDirectLightAccum[launchIndex] = float4(1.0f, 1.0f, 1.0f, 0.0f);
        return;
    }

    const ExtraParams extraParams = extraParamsFromRender(uint(instanceId));
    const float3 rayDirection = gViewDirection[launchIndex].xyz;
    const float3 position = gShadingPosition[launchIndex].xyz;
    const float3 normal = gShadingNormal[launchIndex].xyz;
    const float3 specular = gShadingSpecular[launchIndex].rgb;

    float3 newDirect = float3(0.0f, 0.0f, 0.0f);
    float historyLength = 0.0f;
    if (RtParams.diReproject) {
        int2 prevIndex;
        const float historyWeight = reprojectionWeight(launchIndex, normal, prevIndex);
        if (historyWeight > 0.0f) {
            const float4 prevDirectAccum = gPrevDirectLightAccum[prevIndex];
            newDirect = prevDirectAccum.rgb;
            historyLength = prevDirectAccum.a * historyWeight;
        }
    }

    if (RtParams.debugFlags & 0x2) {
        const float NdotL = (RtParams.lightsCount > 0) ? saturate(dot(normal, normalize(SceneLights[0].position - position))) : 0.0f;
        gDirectLightAccum[launchIndex] = float4(NdotL, RtParams.lightsCount / 4.0f, (extraParams.lightGroupMaskBits != 0) ? 1.0f : 0.0f, 1.0f);
        return;
    }

    float3 resDirect = directLightAt(launchIndex, rayDirection, extraParams, position, normal, RtParams.maxLights, true) * cloudShadowAt(position);
    resDirect += extraParams.selfLight;
    resDirect += emissionFactor(gDiffuse[launchIndex].rgb);

    // Eye light.
    const float eyeLightLambertFactor = max(dot(normal, -rayDirection), 0.0f);
    const float3 eyeLightReflected = reflect(rayDirection, normal);
    const float3 eyeLightSpecularFactor = specular * pow(max(saturate(dot(eyeLightReflected, -rayDirection)), 0.0f), extraParams.specularExponent);
    resDirect += (RtParams.eyeLightDiffuseColor.rgb * eyeLightLambertFactor + RtParams.eyeLightSpecularColor.rgb * eyeLightSpecularFactor);

    // The direct light history is kept short so moving shadows don't leave trails behind.
    historyLength = min(historyLength + 1.0f, max(RtParams.directHistoryLength, 1.0f));
    newDirect = lerp(newDirect, resDirect, 1.0f / historyLength);
    gDirectLightAccum[launchIndex] = float4(newDirect, historyLength);
}

// Indirect light.

[shader("raygeneration")]
void IndirectRayGen() {
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchDims = DispatchRaysDimensions().xy;
    const int instanceId = gInstanceId[launchIndex];
    if ((instanceId < 0) || (RtParams.giSamples == 0)) {
        gIndirectLightAccum[launchIndex] = float4(ambientLight(), 0.0f);
        return;
    }

    const float3 rayOrigin = gShadingPosition[launchIndex].xyz;
    const float3 shadingNormal = gShadingNormal[launchIndex].xyz;
    float3 newIndirect = float3(0.0f, 0.0f, 0.0f);
    float historyLength = 0.0f;
    if (RtParams.giReproject) {
        int2 prevIndex;
        const float historyWeight = reprojectionWeight(launchIndex, shadingNormal, prevIndex);
        if (historyWeight > 0.0f) {
            const float4 prevIndirectAccum = gPrevIndirectLightAccum[prevIndex];
            newIndirect = prevIndirectAccum.rgb;
            historyLength = prevIndirectAccum.a * historyWeight;
        }
    }

    const uint blueNoiseMult = max(64 / RtParams.giSamples, 1);
    for (uint s = 0; s < RtParams.giSamples; s++) {
        const float3 rayDirection = normalize(getCosHemisphereSampleBlueNoise(gBlueNoise, launchIndex, FrParams.frameCount + s * blueNoiseMult, shadingNormal));
        HitInfo payload;
        traceSurfaces(rayOrigin, rayDirection, zeroRayDiff(), true, payload);

        float3 resPosition = float3(0.0f, 0.0f, 0.0f);
        float3 resNormal = float3(0.0f, 0.0f, 0.0f);
        float4 resColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
        int resInstanceId = -1;
        const uint hitCount = min(payload.nhits, MAX_HIT_QUERIES);
        for (uint hit = 0; hit < hitCount; hit++) {
            const uint hitBufferIndex = getHitBufferIndex(hit, launchIndex, launchDims);
            const float4 hitColor = gHitColor[hitBufferIndex];
            const float alphaContrib = resColor.a * hitColor.a;
            if (alphaContrib >= EPSILON) {
                const uint renderIndex = gHitInstanceId[hitBufferIndex];
                const ExtraParams extraParams = extraParamsFromRender(renderIndex);
                resColor.rgb += SrgbToLinear(hitColor.rgb) * alphaContrib;
                resColor.a *= (1.0f - hitColor.a);
                if (resInstanceId < 0) {
                    resPosition = hitPosition(rayOrigin, rayDirection, gHitVelocityDistance[hitBufferIndex].w, extraParams);
                    resNormal = normalize(gHitNormalFog[hitBufferIndex].xyz);
                    resInstanceId = int(renderIndex);
                }
            }

            if (resColor.a <= EPSILON) {
                break;
            }
        }

        // The ambient light is occluded by the surfaces found close to the shading point.
        float ambientVisibility = 1.0f;
        if ((resInstanceId >= 0) && (RtParams.aoRadius > 0.0f)) {
            const float hitDistance = length(resPosition - rayOrigin);
            ambientVisibility = lerp(1.0f - RtParams.aoStrength, 1.0f, smoothstep(0.0f, 1.0f, hitDistance / RtParams.aoRadius));
        }

        float3 resIndirect = RtParams.ambientBaseColor.rgb * ambientVisibility;
        if (resInstanceId >= 0) {
            const ExtraParams extraParams = extraParamsFromRender(uint(resInstanceId));
            const float3 directLight = directLightAt(launchIndex, rayDirection, extraParams, resPosition, resNormal, 1, true) + extraParams.selfLight;
            resIndirect += resColor.rgb * (ambientLight() + directLight + emissionFactor(resColor.rgb)) * RtParams.giDiffuseStrength;
        }

        // The background is desaturated as it usually has a much stronger tint than the light it should represent.
        const float3 environment = sampleEnvironment(rayDirection, RtParams.skyGI, false);
        const float environmentLuma = dot(environment, float3(0.2126f, 0.7152f, 0.0722f));
        resIndirect += lerp(environment, environmentLuma.xxx, 0.5f) * RtParams.giBackgroundStrength * resColor.a;

        historyLength = min(historyLength + 1.0f, 64.0f);
        newIndirect = lerp(newIndirect, resIndirect, 1.0f / historyLength);
    }

    gIndirectLightAccum[launchIndex] = float4(newIndirect, historyLength);
}

// Reflections.

[shader("raygeneration")]
void ReflectionRayGen() {
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchDims = DispatchRaysDimensions().xy;
    const int instanceId = gInstanceId[launchIndex];
    const float reflectionAlpha = gReflection[launchIndex].a;
    if ((instanceId < 0) || (reflectionAlpha <= EPSILON)) {
        return;
    }

    const float3 shadingPosition = gShadingPosition[launchIndex].xyz;
    const float3 viewDirection = gViewDirection[launchIndex].xyz;
    const float3 shadingNormal = gShadingNormal[launchIndex].xyz;
    const float3 rayDirection = reflect(viewDirection, shadingNormal);
    float newReflectionAlpha = 0.0f;

    HitInfo payload;
    traceSurfaces(shadingPosition, rayDirection, zeroRayDiff(), true, payload);

    float3 resPosition = float3(0.0f, 0.0f, 0.0f);
    float3 resNormal = float3(0.0f, 0.0f, 0.0f);
    int resInstanceId = -1;
    float4 resColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float3 resTransparent = float3(0.0f, 0.0f, 0.0f);
    const uint hitCount = min(payload.nhits, MAX_HIT_QUERIES);
    for (uint hit = 0; hit < hitCount; hit++) {
        const uint hitBufferIndex = getHitBufferIndex(hit, launchIndex, launchDims);
        const float4 hitColor = gHitColor[hitBufferIndex];
        float alphaContrib = resColor.a * hitColor.a;
        if (alphaContrib >= EPSILON) {
            const uint hitRenderIndex = gHitInstanceId[hitBufferIndex];
            const ExtraParams hitExtraParams = extraParamsFromRender(hitRenderIndex);
            const bool usesLighting = (hitExtraParams.lightGroupMaskBits > 0);
            const float4 hitNormalFog = gHitNormalFog[hitBufferIndex];
            const float3 vertexPosition = hitPosition(shadingPosition, rayDirection, gHitVelocityDistance[hitBufferIndex].w, hitExtraParams);
            const float3 vertexNormal = normalize(hitNormalFog.xyz);
            if (hitNormalFog.w > EPSILON) {
                const float4 fog = computeFog(hitRenderIndex, hitNormalFog.w);
                resTransparent += fog.rgb * fog.a * alphaContrib;
                alphaContrib *= (1.0f - fog.a);
            }

            if (hitExtraParams.reflectionFactor > EPSILON) {
                const float fresnelAmount = FresnelReflectAmount(vertexNormal, rayDirection, hitExtraParams.reflectionFactor, hitExtraParams.reflectionFresnelFactor);
                newReflectionAlpha += fresnelAmount * alphaContrib * reflectionAlpha;
            }

            const float3 albedo = SrgbToLinear(hitColor.rgb);
            if (usesLighting) {
                resColor.rgb += albedo * alphaContrib;
            }
            else {
                resTransparent += albedo * alphaContrib * (ambientLight() + hitExtraParams.selfLight);
            }

            if (resInstanceId < 0) {
                resPosition = vertexPosition;
                resNormal = vertexNormal;
                resInstanceId = int(hitRenderIndex);
            }

            resColor.a *= (1.0f - hitColor.a);
        }

        if (resColor.a <= EPSILON) {
            break;
        }
    }

    if (resInstanceId >= 0) {
        const ExtraParams extraParams = extraParamsFromRender(uint(resInstanceId));
        const float3 directLight = directLightAt(launchIndex, rayDirection, extraParams, resPosition, resNormal, 1, true) + extraParams.selfLight;
        resColor.rgb *= (ambientLight() + directLight);
        gShadingPosition[launchIndex] = float4(resPosition, 0.0f);
        gViewDirection[launchIndex] = float4(rayDirection, 0.0f);
        gShadingNormal[launchIndex] = float4(resNormal, 0.0f);
        gInstanceId[launchIndex] = resInstanceId;
    }
    else {
        gInstanceId[launchIndex] = -1;
    }

    resColor.rgb += sampleEnvironment(rayDirection, 1.0f, true) * resColor.a + resTransparent;
    gReflection[launchIndex].rgb += LinearToSrgb(max(resColor.rgb, 0.0f)) * reflectionAlpha * saturate(1.0f - newReflectionAlpha);
    gReflection[launchIndex].a = saturate(newReflectionAlpha);
}

// Refractions.

[shader("raygeneration")]
void RefractionRayGen() {
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchDims = DispatchRaysDimensions().xy;
    const int instanceId = gInstanceId[launchIndex];
    const float refractionAlpha = gRefraction[launchIndex].a;
    if ((instanceId < 0) || (refractionAlpha <= EPSILON)) {
        return;
    }

    const ExtraParams extraParams = extraParamsFromRender(uint(instanceId));
    const float3 rayOrigin = gShadingPosition[launchIndex].xyz;
    const float3 viewDirection = gViewDirection[launchIndex].xyz;
    const float3 shadingNormal = gShadingNormal[launchIndex].xyz;
    float3 rayDirection = refract(viewDirection, shadingNormal, extraParams.refractionFactor);
    if (dot(rayDirection, rayDirection) < EPSILON) {
        rayDirection = viewDirection;
    }

    HitInfo payload;
    traceSurfaces(rayOrigin, rayDirection, zeroRayDiff(), true, payload);

    float3 resPosition = float3(0.0f, 0.0f, 0.0f);
    float3 resNormal = float3(0.0f, 0.0f, 0.0f);
    int resInstanceId = -1;
    float4 resColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float3 resTransparent = float3(0.0f, 0.0f, 0.0f);
    const uint hitCount = min(payload.nhits, MAX_HIT_QUERIES);
    for (uint hit = 0; hit < hitCount; hit++) {
        const uint hitBufferIndex = getHitBufferIndex(hit, launchIndex, launchDims);
        const float4 hitColor = gHitColor[hitBufferIndex];
        float alphaContrib = resColor.a * hitColor.a;
        if (alphaContrib >= EPSILON) {
            const uint hitRenderIndex = gHitInstanceId[hitBufferIndex];
            const ExtraParams hitExtraParams = extraParamsFromRender(hitRenderIndex);
            const bool usesLighting = (hitExtraParams.lightGroupMaskBits > 0);
            const float4 hitNormalFog = gHitNormalFog[hitBufferIndex];
            const float3 vertexPosition = hitPosition(rayOrigin, rayDirection, gHitVelocityDistance[hitBufferIndex].w, hitExtraParams);
            if (hitNormalFog.w > EPSILON) {
                const float4 fog = computeFog(hitRenderIndex, hitNormalFog.w);
                resTransparent += fog.rgb * fog.a * alphaContrib;
                alphaContrib *= (1.0f - fog.a);
            }

            const float3 albedo = SrgbToLinear(hitColor.rgb);
            if (usesLighting) {
                resColor.rgb += albedo * alphaContrib;
                if (resInstanceId < 0) {
                    resPosition = vertexPosition;
                    resNormal = normalize(hitNormalFog.xyz);
                    resInstanceId = int(hitRenderIndex);
                }
            }
            else {
                resTransparent += albedo * alphaContrib * (ambientLight() + hitExtraParams.selfLight);
            }

            resColor.a *= (1.0f - hitColor.a);
        }

        if (resColor.a <= EPSILON) {
            break;
        }
    }

    if (resInstanceId >= 0) {
        const ExtraParams hitExtraParams = extraParamsFromRender(uint(resInstanceId));
        const float3 directLight = directLightAt(launchIndex, rayDirection, hitExtraParams, resPosition, resNormal, 1, true) + hitExtraParams.selfLight;
        resColor.rgb *= (ambientLight() + directLight);
    }

    const float2 screenUV = (float2(launchIndex) + 0.5f) / float2(launchDims);
    resColor.rgb += sampleBackground(screenUV) * resColor.a + resTransparent;
    gRefraction[launchIndex].rgb += LinearToSrgb(max(resColor.rgb, 0.0f)) * refractionAlpha;
}
