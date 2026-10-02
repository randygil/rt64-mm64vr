//
// RT64
//

// Bloom for the raytraced output. The bright parts of the image are extracted into a quarter resolution texture
// that is blurred with a separable gaussian filter and added on top of the image by the post processing pass.

#define BLOCK_SIZE 8

struct BloomCB {
    uint2 OutputSize;
    float2 InputTexelSize;
    uint Mode;
    float Threshold;
};

[[vk::push_constant]] ConstantBuffer<BloomCB> gConstants : register(b0);
Texture2D<float4> gInput : register(t1);
[[vk::image_format("rgba16f")]] RWTexture2D<float4> gOutput : register(u2);
SamplerState gSampler : register(s3);

static const uint ModePrefilter = 0;
static const uint ModeBlurHorizontal = 1;

float3 sanitize(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0.0f, 0.0f, 0.0f) : max(c, 0.0f);
}

// Keeps the part of the color above the threshold with a soft transition.
float3 brightPass(float3 color) {
    const float Knee = 0.25f;
    const float brightness = max(color.r, max(color.g, color.b));
    float soft = clamp(brightness - gConstants.Threshold + Knee, 0.0f, 2.0f * Knee);
    soft = (soft * soft) / (4.0f * Knee + 1e-4f);
    const float contribution = max(soft, brightness - gConstants.Threshold) / max(brightness, 1e-4f);
    return color * contribution;
}

[numthreads(BLOCK_SIZE, BLOCK_SIZE, 1)]
void CSMain(uint2 pixel : SV_DispatchThreadID) {
    if (any(pixel >= gConstants.OutputSize)) {
        return;
    }

    const float2 uv = (float2(pixel) + 0.5f) / float2(gConstants.OutputSize);
    if (gConstants.Mode == ModePrefilter) {
        // Four bilinear samples cover the 4x4 block of the input that corresponds to this pixel.
        const float2 offset = gConstants.InputTexelSize;
        float3 sum = float3(0.0f, 0.0f, 0.0f);
        sum += brightPass(sanitize(gInput.SampleLevel(gSampler, uv + float2(-offset.x, -offset.y), 0).rgb));
        sum += brightPass(sanitize(gInput.SampleLevel(gSampler, uv + float2(offset.x, -offset.y), 0).rgb));
        sum += brightPass(sanitize(gInput.SampleLevel(gSampler, uv + float2(-offset.x, offset.y), 0).rgb));
        sum += brightPass(sanitize(gInput.SampleLevel(gSampler, uv + float2(offset.x, offset.y), 0).rgb));
        gOutput[pixel] = float4(min(sum * 0.25f, 16.0f), 1.0f);
        return;
    }

    // Gaussian blur with a wide radius. The bilinear filter is used to take two texels per sample.
    const float Weights[5] = { 0.1531f, 0.2779f, 0.1395f, 0.0386f, 0.0057f };
    const float Offsets[5] = { 0.0f, 1.4211f, 3.3289f, 5.2392f, 7.1575f };
    const float2 direction = (gConstants.Mode == ModeBlurHorizontal) ? float2(gConstants.InputTexelSize.x, 0.0f) : float2(0.0f, gConstants.InputTexelSize.y);
    float3 sum = gInput.SampleLevel(gSampler, uv, 0).rgb * Weights[0];
    float weightSum = Weights[0];
    for (uint i = 1; i < 5; i++) {
        sum += gInput.SampleLevel(gSampler, uv + direction * Offsets[i], 0).rgb * Weights[i];
        sum += gInput.SampleLevel(gSampler, uv - direction * Offsets[i], 0).rgb * Weights[i];
        weightSum += Weights[i] * 2.0f;
    }

    gOutput[pixel] = float4(sum / weightSum, 1.0f);
}
