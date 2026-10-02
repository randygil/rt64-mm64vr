//
// RT64
//

// Temporal anti-aliasing for the raytraced output. The primary rays are jittered every frame and the result is
// accumulated with the reprojected history, which is clipped to the neighborhood of the current pixel to avoid ghosting.

#define BLOCK_SIZE 8

struct TemporalAACB {
    uint2 TextureSize;
    float2 TexelSize;
    float BlendFactor;
    uint Reset;
};

[[vk::push_constant]] ConstantBuffer<TemporalAACB> gConstants : register(b0);
Texture2D<float4> gCurrent : register(t1);
Texture2D<float4> gHistory : register(t2);
Texture2D<float2> gFlow : register(t3);
[[vk::image_format("rgba16f")]] RWTexture2D<float4> gOutput : register(u4);
SamplerState gSampler : register(s5);

float3 RGBToYCoCg(float3 c) {
    return float3(
        dot(c, float3(0.25f, 0.5f, 0.25f)),
        dot(c, float3(0.5f, 0.0f, -0.5f)),
        dot(c, float3(-0.25f, 0.5f, -0.25f))
    );
}

float3 YCoCgToRGB(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Invalid values must never reach the history, as the neighborhood statistics would spread them over the whole image.
float3 sanitize(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0.0f, 0.0f, 0.0f) : max(c, 0.0f);
}

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void CSMain(uint2 pixel : SV_DispatchThreadID) {
    if (any(pixel >= gConstants.TextureSize)) {
        return;
    }

    const float3 current = sanitize(gCurrent.Load(int3(pixel, 0)).rgb);
    if (gConstants.Reset) {
        gOutput[pixel] = float4(current, 1.0f);
        return;
    }

    // Statistics of the neighborhood used to clip the history.
    const int2 maxPixel = int2(gConstants.TextureSize) - 1;
    float3 m1 = float3(0.0f, 0.0f, 0.0f);
    float3 m2 = float3(0.0f, 0.0f, 0.0f);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            const int2 samplePixel = clamp(int2(pixel) + int2(x, y), int2(0, 0), maxPixel);
            const float3 c = RGBToYCoCg(sanitize(gCurrent.Load(int3(samplePixel, 0)).rgb));
            m1 += c;
            m2 += c * c;
        }
    }

    const float3 mean = m1 / 9.0f;
    const float3 sigma = sqrt(max(m2 / 9.0f - mean * mean, 0.0f));
    const float3 boxMin = mean - sigma * 1.25f;
    const float3 boxMax = mean + sigma * 1.25f;

    // The flow points from the current pixel to its position in the previous frame.
    float2 flow = gFlow.Load(int3(pixel, 0));
    if (any(isnan(flow)) || any(isinf(flow))) {
        flow = float2(0.0f, 0.0f);
    }

    const float2 prevUV = (float2(pixel) + 0.5f + flow) * gConstants.TexelSize;
    if (any(prevUV < 0.0f) || any(prevUV > 1.0f)) {
        gOutput[pixel] = float4(current, 1.0f);
        return;
    }

    float3 history = sanitize(gHistory.SampleLevel(gSampler, prevUV, 0).rgb);
    history = YCoCgToRGB(clamp(RGBToYCoCg(history), boxMin, boxMax));
    gOutput[pixel] = float4(lerp(history, current, gConstants.BlendFactor), 1.0f);
}
