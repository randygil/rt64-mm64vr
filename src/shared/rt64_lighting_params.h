//
// RT64
//
// Parameters of the enhanced raster lighting. Everything is expressed in the space the geometry is drawn in (RT64's
// world space), so the shaders don't need to know anything about the game. Matrices follow the same convention as the
// rest of RT64: mul(matrix, vector) in the shaders is the row vector product done on the CPU with hlslpp.
//

#pragma once

#include "shared/rt64_hlsl.h"

// Flags of LightingParams::settings.w.
#define LIGHTING_SCENE_FLAG_SKY_HIDDEN  0x1
#define LIGHTING_SCENE_FLAG_SKY_TINT    0x2
#define LIGHTING_SCENE_FLAG_GBUFFER     0x4

// Flags of the draw calls drawn into the normal buffer.
#define LIGHTING_GBUFFER_ALPHA_TESTED   0x1
#define LIGHTING_GBUFFER_RSP_LIT        0x2
#define LIGHTING_GBUFFER_FOLIAGE        0x4
#define LIGHTING_GBUFFER_BUMP           0x8

#ifdef HLSL_CPU
namespace interop {
#endif
    struct LightingParams {
        float4x4 viewProj;
        float4x4 invViewProj;
        float4x4 shadowMatrix;

        // Converts a pixel position of the framebuffer (pixel centers at .5) into the clip space of the projection:
        // clip.xy = pixel * pixelToClip.xy + pixelToClip.zw.
        float4 pixelToClip;

        // Converts a depth buffer value into the clip space depth (clip.z = depth * x + y). Z is the depth from which a
        // pixel is considered background (nothing was drawn on it).
        float4 depthToClip;

        // Region of the framebuffer covered by the projection in pixels (left, top, right, bottom).
        float4 viewportRect;

        // Direction towards the sun (w = 1 if there's a sun).
        float4 sunDirection;

        // Color and intensity of the sun.
        float4 sunColor;

        // Light coming from the sky and the surroundings.
        float4 ambientColor;

        // Light received by surfaces facing down, from the ground.
        float4 groundColor;

        // Basis of the world (directions pointing right, up and forward) and its origin (w = 1 if known), in the space
        // of the geometry. Effects that must stay fixed in the world (like clouds) use them.
        float4 worldRight;
        float4 worldUp;
        float4 worldForward;
        float4 worldOrigin;

        // Position of the camera.
        float4 cameraPosition;

        // Light carried close to the camera for scenes without a sun (xyz position, w radius; w = 0 if disabled) and its
        // color (w exponent of the falloff).
        float4 pointLightPosition;
        float4 pointLightColor;

        // Fog of the game in the same terms as the RSP: alpha = (clip.z / clip.w) * mul + offset, in 0-255 (x mul, y
        // offset, z = 1 if enabled). Lighting fades out with it.
        float4 fog;

        // x size of a shadow map texel in world units, y constant depth bias, z normal offset in texels, w softness in texels.
        float4 shadowParams;

        // x inverse width, y inverse height of the shadow map, z strength of the shadows, w depth range of the shadow map in
        // world units.
        float4 shadowMapParams;

        // Shadows that get softer away from what casts them (PCSS): x tangent of the angular radius of the sun (0 keeps the
        // fixed softness of shadowParams), y least and z most softness in texels.
        float4 shadowParams2;

        // x overall strength, y exposure, z wrap of the sun's diffuse term, w strength of the shading from the normals.
        float4 lightingParams;

        // Foliage: x wrap of the diffuse term, y light passing through the leaves, z darkest shadow, w distance the shadow
        // lookups are moved towards the sun relative to the size of the tree.
        float4 foliageParams;

        // Ambient occlusion: x radius in world units, y strength, z power, w slices (0 if disabled).
        float4 aoParams;

        // x strength of the ambient occlusion on the direct light, y strength on foliage, zw size of the ambient occlusion
        // texture.
        float4 aoParams2;

        // Contact shadows: x length of the rays towards the light in world units, y thickness given to the surfaces in the
        // depth buffer, z strength, w steps of the rays (0 if disabled).
        float4 contactParams;

        // x strength of the tint the light takes from skies that aren't daytime skies (LIGHTING_SCENE_FLAG_SKY_TINT), y height
        // above a surface under which the ambient occlusion ignores other surfaces (thin layers like signs and posters).
        float4 miscParams;

        // Shadows of the clouds of the procedural sky on the ground: x inverse of the height of the clouds in world units,
        // y cells of their noise per cloud height, z coverage of the clouds and w strength of the shadows (0 if disabled).
        float4 cloudShadowParams;

        // xy offset of the shapes of the clouds and zw offset of the noise that warps them, in cells of each noise (they
        // move with the wind, like the clouds of the sky).
        float4 cloudShadowOffset;

        // x warp of the shapes, y softness of the edges of the shadows, z octaves of the noise (0 skips the warp).
        float4 cloudShadowMisc;

        // Shadows of the light carried in scenes without a sun (pointLightPosition), from the six faces of a cube laid out
        // in a 3x2 atlas: x and y turn the distance along a face's axis into its depth (depth = x + y / distance), z size of
        // a face in texels and w strength (0 if disabled).
        float4 pointShadowParams;

        // xyz position the shadows were drawn from (the same for every scene of the frame), w distance of the near plane.
        float4 pointShadowPosition;

        // x normal offset in texels, y depth bias in texels and z softness in texels.
        float4 pointShadowParams2;

        // Glowing surfaces of scenes without a sun (see lightingEmissiveMask): x brightness of the strongest channel from
        // which colorful surfaces glow, y brightness added to the glowing surfaces themselves, z strength of the light they cast around them and
        // w most light they can add to a surface (0 if disabled).
        float4 emissiveParams;

        // x debug view, y quality level, z sample count of the depth buffer, w flags.
        uint4 settings;
    };

    struct LightingShadowCB {
        float4x4 shadowMatrix;
        uint renderIndex;
        uint3 padding;
    };

    struct LightingGBufferCB {
        float4 cameraPosition;
        float2 screenScale;
        float2 screenOffset;
        uint renderIndex;
        uint indexStart;
        uint flags;
        float bumpStrength;
    };

    struct LightingAOCB {
        uint sceneIndex;
        uint2 outputSize;
        uint frameIndex;
    };

    struct LightingAOBlurCB {
        uint2 size;
        int2 direction;
    };

    struct LightingEmissiveCB {
        uint sceneIndex;
        uint2 outputSize;
        uint padding;
    };

    struct LightingEmissiveBlurCB {
        uint2 size;
        int2 direction;
        uint radius;
        uint3 padding;
    };

    struct LightingComposeCB {
        uint sceneIndex;

        // With multisampling, the nearest surface of each pixel is lit first (0) and the samples of a farther one at the
        // edges of objects afterwards (1), so each surface of an edge pixel gets its own light. The low quality preset
        // lights the nearest surface for all the samples at once instead (2).
        uint surfacePass;
        uint2 padding;
    };
#ifdef HLSL_CPU
};
#endif
