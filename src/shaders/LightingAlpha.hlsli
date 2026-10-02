//
// RT64
//
// Coverage of a draw call's pixel as the RDP would compute it, for the passes of the enhanced lighting that draw the
// scene again with simpler shaders (shadow maps, normals). Only the alpha matters: the textures and the color combiner
// run exactly like in the raster shaders, so cutouts like leaves keep their shape.
//

#pragma once

#include "shared/rt64_color_combiner.h"

#include "FbRendererCommon.hlsli"
#include "TextureSampler.hlsli"

float lightingCoverageAlpha(uint renderIndex, float2 vertexUV, float4 vertexColor, out bool alphaCompareFailed) {
    const uint instanceIndex = instanceRenderIndices[renderIndex].instanceIndex;
    const RenderParams rp = DynamicRenderParams[instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };
    const float ddxuvx = ddx(vertexUV.x);
    const float ddyuvy = ddy(vertexUV.y);
    int tileIndex0 = 0;
    int tileIndex1 = 1;
    float lodFraction;
    computeLOD(otherMode, instanceRenderIndices[renderIndex].rdpTileCount, instanceRDPParams[instanceIndex].primLOD, 1.0f, ddxuvx, ddyuvy, tileIndex0, tileIndex1, lodFraction);

    float4 texVal0 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 texVal1 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (renderFlagUsesTexture0(rp.flags)) {
        const uint globalTileIndex = instanceRenderIndices[renderIndex].rdpTileIndex + tileIndex0;
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = renderCMS0(rp.flags);
            rdpTile.cmt = renderCMT0(rp.flags);
            rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
        }

        texVal0 = sampleTexture(otherMode, rp.flags, vertexUV, ddx(vertexUV), ddy(vertexUV), rdpTile, GPUTiles[globalTileIndex], false);
    }

    if (renderFlagUsesTexture1(rp.flags)) {
        const bool oneCycleHardwareBug = (otherMode.cycleType() == G_CYC_1CYCLE);
        const uint globalTileIndex = instanceRenderIndices[renderIndex].rdpTileIndex + (oneCycleHardwareBug ? tileIndex0 : tileIndex1);
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = oneCycleHardwareBug ? renderCMS0(rp.flags) : renderCMS1(rp.flags);
            rdpTile.cmt = oneCycleHardwareBug ? renderCMT0(rp.flags) : renderCMT1(rp.flags);
            rdpTile.nativeSampler = oneCycleHardwareBug ? renderFlagNativeSampler0(rp.flags) : renderFlagNativeSampler1(rp.flags);
        }

        texVal1 = sampleTexture(otherMode, rp.flags, vertexUV, ddx(vertexUV), ddy(vertexUV), rdpTile, GPUTiles[globalTileIndex], oneCycleHardwareBug);
    }

    float4 combinerColor;
    float alphaCompareValue;
    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texVal0;
    ccInputs.texVal1 = texVal1;
    ccInputs.primColor = instanceRDPParams[instanceIndex].primColor;
    ccInputs.shadeColor = vertexColor;
    ccInputs.envColor = instanceRDPParams[instanceIndex].envColor;
    ccInputs.keyCenter = instanceRDPParams[instanceIndex].keyCenter;
    ccInputs.keyScale = instanceRDPParams[instanceIndex].keyScale;
    ccInputs.lodFraction = lodFraction;
    ccInputs.primLodFrac = instanceRDPParams[instanceIndex].primLOD.x;
    ccInputs.noise = 0.5f;
    ccInputs.K4 = (instanceRDPParams[instanceIndex].convertK[4] / 255.0f);
    ccInputs.K5 = (instanceRDPParams[instanceIndex].convertK[5] / 255.0f);
    colorCombiner.run(ccInputs, combinerColor, alphaCompareValue);

    // Dithered alpha compare is resolved with a fixed threshold so the result doesn't flicker.
    alphaCompareFailed = false;
    if (otherMode.alphaCompare() == G_AC_DITHER) {
        alphaCompareFailed = (alphaCompareValue < 0.5f);
    }
    else if (otherMode.alphaCompare() == G_AC_THRESHOLD) {
        alphaCompareFailed = (alphaCompareValue < instanceRDPParams[instanceIndex].blendColor.a);
    }

    return otherMode.cvgXAlpha() ? combinerColor.a : 1.0f;
}

// Whether the pixel of the draw call would be discarded. Coverage below the threshold is considered empty.
bool lightingPixelDiscarded(uint renderIndex, float2 vertexUV, float4 vertexColor, float coverageThreshold) {
    bool alphaCompareFailed;
    const float coverage = lightingCoverageAlpha(renderIndex, vertexUV, vertexColor, alphaCompareFailed);
    return alphaCompareFailed || (coverage < coverageThreshold);
}
