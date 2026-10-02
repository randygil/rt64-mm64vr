//
// RT64
//

#include "Color.hlsli"
#include "Constants.hlsli"
#include "Math.hlsli"

SamplerState gSampler : register(s0);
Texture2D<float4> gFlow : register(t1);
Texture2D<float4> gDiffuse : register(t2);
Texture2D<float4> gDirectLight : register(t3);
Texture2D<float4> gIndirectLight : register(t4);
Texture2D<float4> gReflection : register(t5);
Texture2D<float4> gRefraction : register(t6);
Texture2D<float4> gTransparent : register(t7);

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    // The diffuse buffer stores the average albedo of the lit surfaces and their coverage of the pixel. The rest of the
    // buffers are already in sRGB space and were weighted by their own coverage.
    float4 diffuse = gDiffuse.SampleLevel(gSampler, uv, 0);
    float3 directLight = gDirectLight.SampleLevel(gSampler, uv, 0).rgb;
    float3 indirectLight = gIndirectLight.SampleLevel(gSampler, uv, 0).rgb;
    float3 reflection = gReflection.SampleLevel(gSampler, uv, 0).rgb;
    float3 refraction = gRefraction.SampleLevel(gSampler, uv, 0).rgb;
    float3 transparent = gTransparent.SampleLevel(gSampler, uv, 0).rgb;
    float3 result = transparent + reflection + refraction;
    if (diffuse.a > EPSILON) {
        result += LinearToSrgb(max(diffuse.rgb * (directLight + indirectLight), 0.0f)) * diffuse.a;
    }

    return float4(result, 1.0f);
}
