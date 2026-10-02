//
// RT64
//
// Copies the color target into a texture that can be sampled while the target is drawn to, averaging the samples of
// multisampled targets.
//

#ifdef MULTISAMPLING
Texture2DMS<float4> gInput : register(t1, space0);
#else
Texture2D<float4> gInput : register(t1, space0);
#endif

float4 PSMain(in float4 pixelPosition : SV_POSITION) : SV_TARGET {
    const int2 pixel = int2(pixelPosition.xy);
#ifdef MULTISAMPLING
    uint width, height, sampleCount;
    gInput.GetDimensions(width, height, sampleCount);
    float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint i = 0; i < sampleCount; i++) {
        color += gInput.Load(pixel, i);
    }

    return color / float(sampleCount);
#else
    return gInput.Load(int3(pixel, 0));
#endif
}
