//
// RT64
//
// Sky-view lookup table of the procedural sky: the sunlight scattered once towards the observer by an Earth-like
// atmosphere (Rayleigh and Mie scattering, ozone absorption) plus a cheap isotropic term for the light scattered more
// than once, for every direction above the horizon relative to the sun. It's graded here (exposure, range, saturation
// and tint) so the sky pass only needs one lookup. It only depends on the elevation of the sun and the settings, so
// the CPU only renders it again when they change.
//

#include "LightingSkyCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<LightingSkyLutCB> gConstants : register(b0, space0);

static const float PlanetRadius = 6360e3f;
static const float AtmosphereRadius = 6460e3f;
static const float3 RayleighScattering = float3(5.802e-6f, 13.558e-6f, 33.1e-6f);
static const float RayleighHeight = 8e3f;
static const float MieScattering = 3.996e-6f;
static const float MieExtinction = 4.44e-6f;
static const float MieHeight = 1.2e3f;
static const float MieAnisotropy = 0.8f;
static const float3 OzoneAbsorption = float3(0.650e-6f, 1.881e-6f, 0.085e-6f);
static const float MaxViewDistance = 400e3f;
static const uint ViewSteps = 24;
static const uint SunSteps = 6;

// Distance from a point inside a sphere centered at the origin to where the ray leaves it.
float sphereExitDistance(float3 origin, float3 direction, float radius) {
    const float b = dot(origin, direction);
    const float r = length(origin);
    const float c = (r - radius) * (r + radius);
    const float root = sqrt(max(b * b - c, 0.0f));

    // Avoids the cancellation of -b + root when b is positive.
    return (b > 0.0f) ? (-c / (b + root)) : (root - b);
}

bool rayHitsPlanet(float3 origin, float3 direction) {
    const float b = dot(origin, direction);
    const float r = length(origin);
    const float c = (r - PlanetRadius) * (r + PlanetRadius);
    return (b < 0.0f) && ((b * b - c) >= 0.0f);
}

// Density of the air molecules (x), the aerosols (y) and the ozone (z) at an altitude.
float3 atmosphereDensities(float altitude) {
    const float rayleigh = exp(-altitude / RayleighHeight);
    const float mie = exp(-altitude / MieHeight) * gConstants.atmosphere.x;
    const float ozone = max(1.0f - abs(altitude - 25e3f) / 15e3f, 0.0f) * gConstants.atmosphere.z;
    return float3(rayleigh, mie, ozone);
}

float3 atmosphereExtinction(float3 densities) {
    return RayleighScattering * densities.x + MieExtinction * densities.y + OzoneAbsorption * densities.z;
}

// Fraction of the sunlight that reaches a point.
float3 sunTransmittance(float3 position, float3 sunDirection) {
    if (rayHitsPlanet(position, sunDirection)) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    const float stepSize = sphereExitDistance(position, sunDirection, AtmosphereRadius) / float(SunSteps);
    float3 opticalDepth = float3(0.0f, 0.0f, 0.0f);
    for (uint i = 0; i < SunSteps; i++) {
        const float3 samplePosition = position + sunDirection * (stepSize * (float(i) + 0.5f));
        const float altitude = max(length(samplePosition) - PlanetRadius, 0.0f);
        opticalDepth += atmosphereExtinction(atmosphereDensities(altitude)) * stepSize;
    }

    return exp(-opticalDepth);
}

float phaseRayleigh(float cosine) {
    return 3.0f / (16.0f * LIGHTING_SKY_PI) * (1.0f + cosine * cosine);
}

// Cornette-Shanks approximation of the Mie phase function.
float phaseMie(float cosine, float g) {
    const float g2 = g * g;
    const float denominator = (2.0f + g2) * pow(max(1.0f + g2 - 2.0f * g * cosine, 1e-6f), 1.5f);
    return 3.0f / (8.0f * LIGHTING_SKY_PI) * ((1.0f - g2) * (1.0f + cosine * cosine)) / denominator;
}

float4 PSMain(in float4 pixelPosition : SV_POSITION) : SV_TARGET {
    const float3 direction = lightingSkyLutDirection(uint2(pixelPosition.xy));
    const float3 sunDirection = gConstants.sunDirection.xyz;
    const float3 origin = float3(0.0f, PlanetRadius + gConstants.observer.x, 0.0f);
    const float pathLength = min(sphereExitDistance(origin, direction, AtmosphereRadius), MaxViewDistance);
    float3 rayleigh = float3(0.0f, 0.0f, 0.0f);
    float3 mie = float3(0.0f, 0.0f, 0.0f);
    float3 scattered = float3(0.0f, 0.0f, 0.0f);
    float3 opticalDepth = float3(0.0f, 0.0f, 0.0f);
    float previousDistance = 0.0f;
    for (uint i = 0; i < ViewSteps; i++) {
        // The steps get longer with the distance, where the light that reaches the observer is already dimmed.
        const float t = float(i + 1) / float(ViewSteps);
        const float distance = pathLength * t * t;
        const float stepSize = distance - previousDistance;
        const float3 samplePosition = origin + direction * (0.5f * (distance + previousDistance));
        previousDistance = distance;

        const float altitude = max(length(samplePosition) - PlanetRadius, 0.0f);
        const float3 densities = atmosphereDensities(altitude);
        const float3 extinction = atmosphereExtinction(densities);
        const float3 viewTransmittance = exp(-(opticalDepth + extinction * (0.5f * stepSize)));
        opticalDepth += extinction * stepSize;

        const float3 sunLight = sunTransmittance(samplePosition, sunDirection) * viewTransmittance * stepSize;
        const float3 rayleighScattering = RayleighScattering * densities.x;
        const float mieScattering = MieScattering * densities.y;
        rayleigh += rayleighScattering * sunLight;
        mie += mieScattering * sunLight;
        scattered += (rayleighScattering + mieScattering) * viewTransmittance * stepSize;
    }

    const float cosine = dot(direction, sunDirection);
    float3 color = rayleigh * phaseRayleigh(cosine) + mie * phaseMie(cosine, MieAnisotropy);

    // The light scattered more than once fills the sky evenly, the more the higher the sun is.
    const float3 sunGround = sunTransmittance(origin, sunDirection);
    color += scattered * sunGround * (gConstants.atmosphere.y / (4.0f * LIGHTING_SKY_PI)) * max(sunDirection.y + 0.1f, 0.0f);

    // Exposure, then a compression of the range between the zenith and the horizon that keeps the hue, saturation and tint.
    const float3 LumaWeights = float3(0.2126f, 0.7152f, 0.0722f);
    color *= gConstants.sunDirection.w;
    color *= pow(max(dot(color, LumaWeights), 1e-6f), gConstants.atmosphere.w - 1.0f);
    const float luma = dot(color, LumaWeights);
    color = max(lerp(float3(luma, luma, luma), color, gConstants.skyTint.w), 0.0f) * gConstants.skyTint.rgb;
    return float4(color, 1.0f);
}
