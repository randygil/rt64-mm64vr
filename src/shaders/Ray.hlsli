//
// RT64
//

#pragma once

#define RAY_MIN_DISTANCE                    0.1f
#define RAY_MAX_DISTANCE                    10000000.0f

struct RayDiff {
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;
};

// Rays that accept opaque hits stop looking for surfaces behind them, which is faster but can skip coplanar decals.
#define HIT_INFO_FLAG_ACCEPT_OPAQUE 0x1

struct HitInfo {
    uint nhits;
    uint flags;
    RayDiff rayDiff;
};

struct ShadowHitInfo {
    float shadowHit;
    RayDiff rayDiff;
};

RayDiff zeroRayDiff() {
    RayDiff rayDiff;
    rayDiff.dOdx = float3(0.0f, 0.0f, 0.0f);
    rayDiff.dOdy = float3(0.0f, 0.0f, 0.0f);
    rayDiff.dDdx = float3(0.0f, 0.0f, 0.0f);
    rayDiff.dDdy = float3(0.0f, 0.0f, 0.0f);
    return rayDiff;
}

// Ray differentials as described by Igehy in "Tracing Ray Differentials".

RayDiff propagateRayDiffs(RayDiff rayDiff, float3 D, float t, float3 N) {
    float3 dodx = rayDiff.dOdx + t * rayDiff.dDdx;
    float3 dody = rayDiff.dOdy + t * rayDiff.dDdy;
    float DdotN = dot(D, N);
    float rcpDN = (abs(DdotN) > 1e-6f) ? (1.0f / DdotN) : 0.0f;
    float dtdx = -dot(dodx, N) * rcpDN;
    float dtdy = -dot(dody, N) * rcpDN;
    dodx += D * dtdx;
    dody += D * dtdy;

    RayDiff propRayDiff;
    propRayDiff.dOdx = dodx;
    propRayDiff.dOdy = dody;
    propRayDiff.dDdx = rayDiff.dDdx;
    propRayDiff.dDdy = rayDiff.dDdy;
    return propRayDiff;
}

void computeBarycentricDifferentials(RayDiff rayDiff, float3 edge01, float3 edge02, float3 faceNormal, out float2 dBarydx, out float2 dBarydy) {
    float3 Nu = cross(edge02, faceNormal);
    float3 Nv = cross(edge01, faceNormal);
    float NuDot = dot(Nu, edge01);
    float NvDot = dot(Nv, edge02);
    float3 Lu = (abs(NuDot) > 1e-12f) ? (Nu / NuDot) : float3(0.0f, 0.0f, 0.0f);
    float3 Lv = (abs(NvDot) > 1e-12f) ? (Nv / NvDot) : float3(0.0f, 0.0f, 0.0f);
    dBarydx.x = dot(Lu, rayDiff.dOdx);
    dBarydx.y = dot(Lv, rayDiff.dOdx);
    dBarydy.x = dot(Lu, rayDiff.dOdy);
    dBarydy.y = dot(Lv, rayDiff.dOdy);
}

void computeTextureDifferentials(float2 dBarydx, float2 dBarydy, float2 uv0, float2 uv1, float2 uv2, out float2 dUVdx, out float2 dUVdy) {
    float2 uv01 = uv1 - uv0;
    float2 uv02 = uv2 - uv0;
    dUVdx = dBarydx.x * uv01 + dBarydx.y * uv02;
    dUVdy = dBarydy.x * uv01 + dBarydy.y * uv02;
}
