//
// RT64
//
// Enhanced raster lighting: sun shadows and relighting of the opaque surfaces of each 3D projection with classic raster
// techniques (shadow maps and passes over the depth buffer), as a cheap alternative to the path tracer that runs on any
// GPU. It only relies on generic data: the matrices of the projection, the lights estimated by the State and the basis
// of the world provided by the host, so it works the same for any game.
//

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "common/rt64_common.h"
#include "common/rt64_plume.h"
#include "shared/rt64_lighting_params.h"
#include "shared/rt64_point_light.h"

#include "rt64_buffer_uploader.h"
#include "rt64_descriptor_sets.h"
#include "rt64_framebuffer_renderer_call.h"
#include "rt64_lighting_sky.h"
#include "rt64_post_effects.h"
#include "rt64_render_target.h"

namespace RT64 {
    struct ShaderLibrary;

    // Global settings of the enhanced lighting, set by the host. Safe to call from any thread.
    void setRasterLightingEnabled(bool enabled);
    bool isRasterLightingEnabled();

    // 0 low, 1 medium, 2 high, 3 ultra.
    void setRasterLightingQuality(int quality);
    int getRasterLightingQuality();

    // Whether the passes that cost the most on weak GPUs run: the normal buffer, the procedural sky and the post effects.
    // Low leaves them out (standalone VR headsets and integrated GPUs); RT64_LIGHT_LOW_EXTRAS 1 keeps them to compare.
    bool rasterLightingExtrasEnabled();

    struct LightingComposeDescriptorSet : RenderDescriptorSetBase {
        uint32_t gLightingParams;
        uint32_t gDepth;
        uint32_t gShadowMap;
        uint32_t gShadowSampler;
        uint32_t gNormalBuffer;
        uint32_t gAmbientOcclusion;
        uint32_t gSkyAnalysis;
        uint32_t gSceneColor;
        uint32_t gEmissiveLight;
        uint32_t gPointShadowMap;

        LightingComposeDescriptorSet(const RenderSampler *shadowSampler, RenderDevice *device = nullptr) {
            builder.begin();
            gLightingParams = builder.addStructuredBuffer(1);
            gDepth = builder.addTexture(2);
            gShadowMap = builder.addTexture(3);
            gShadowSampler = builder.addImmutableSampler(4, shadowSampler);
            gNormalBuffer = builder.addTexture(5);
            gAmbientOcclusion = builder.addTexture(6);
            gSkyAnalysis = builder.addStructuredBuffer(7);
            gSceneColor = builder.addTexture(8);
            gEmissiveLight = builder.addTexture(9);
            gPointShadowMap = builder.addTexture(10);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct LightingAODescriptorSet : RenderDescriptorSetBase {
        uint32_t gLightingParams;
        uint32_t gDepth;
        uint32_t gNormalBuffer;
        uint32_t gOutput;

        LightingAODescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gLightingParams = builder.addStructuredBuffer(1);
            gDepth = builder.addTexture(2);
            gNormalBuffer = builder.addTexture(3);
            gOutput = builder.addReadWriteTexture(4);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct LightingEmissiveDescriptorSet : RenderDescriptorSetBase {
        uint32_t gLightingParams;
        uint32_t gDepth;
        uint32_t gSceneColor;
        uint32_t gOutput;

        LightingEmissiveDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gLightingParams = builder.addStructuredBuffer(1);
            gDepth = builder.addTexture(2);
            gSceneColor = builder.addTexture(3);
            gOutput = builder.addReadWriteTexture(4);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // Also used by the blur of the light of the glowing surfaces.
    struct LightingAOBlurDescriptorSet : RenderDescriptorSetBase {
        uint32_t gInput;
        uint32_t gOutput;

        LightingAOBlurDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gInput = builder.addTexture(1);
            gOutput = builder.addReadWriteTexture(2);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // Draw call and flags of each triangle (low 24 bits render index, high 8 bits LIGHTING_GBUFFER_* flags), so the
    // passes that draw the scene again can draw many draw calls at once.
    struct LightingTriangleDrawSet : RenderDescriptorSetBase {
        uint32_t gTriangleDraws;

        LightingTriangleDrawSet(RenderDevice *device = nullptr) {
            builder.begin();
            gTriangleDraws = builder.addStructuredBuffer(0);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct LightingCopyDescriptorSet : RenderDescriptorSetBase {
        uint32_t gInput;

        LightingCopyDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gInput = builder.addTexture(1);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // Everything needed to light one projection, in the space the geometry is drawn in.
    struct LightingSceneDesc {
        hlslpp::float4x4 viewProj;
        hlslpp::float4x4 view;

        // Mapping from the pixels of the framebuffer and the depth buffer to the clip space of the projection.
        hlslpp::float4 pixelToClip;
        hlslpp::float4 depthToClip;

        // Basis of the world (right, up, forward) and its origin (w = 1 if known) in the space of the geometry.
        hlslpp::float3 worldRight;
        hlslpp::float3 worldUp;
        hlslpp::float3 worldForward;
        hlslpp::float4 worldOrigin;

        // When the frame is interpolated between two game frames, the basis above follows the interpolated geometry, and
        // these are the basis of the game frame the lights and the focus position were placed with.
        bool worldInterpolated = false;
        hlslpp::float3 gameWorldRight;
        hlslpp::float3 gameWorldUp;
        hlslpp::float3 gameWorldForward;
        hlslpp::float4 gameWorldOrigin;

        // Lights estimated for the projection.
        const interop::PointLight *lights = nullptr;
        uint32_t lightCount = 0;

        // Fog of the game (mul, offset), if any.
        bool fogEnabled = false;
        float fogMul = 0.0f;
        float fogOffset = 0.0f;

        // Transformation applied by the raster vertex shader and its viewport, to draw the surfaces again.
        interop::float2 screenScale;
        interop::float2 screenOffset;
        RenderViewport viewport;

        // The game has a sky the host removed (e.g. in VR), so the background holds nothing worth keeping.
        bool skyHidden = false;

        // Position of the player in the space of the geometry, if the host knows it (w = 1).
        hlslpp::float4 focusPosition = { 0.0f, 0.0f, 0.0f, 0.0f };
    };

    struct LightingRenderer {
        struct GBufferDraw {
            uint32_t instanceIndex;
            uint32_t flags;
        };

        struct Scene {
            interop::LightingParams params;
            RenderRect rect;
            RenderViewport viewport;
            interop::float2 screenScale;
            interop::float2 screenOffset;
            std::vector<GBufferDraw> gbufferDraws;
            RenderTarget *colorTarget = nullptr;
            RenderTarget *depthTarget = nullptr;
            hlslpp::float3 cameraPosition;
            hlslpp::float3 viewDirection;
            float frustumSlope = 1.0f;
            bool hasSun = false;
            hlslpp::float3 sunDirection;
            hlslpp::float3 worldRight;
            hlslpp::float3 worldUp;
            hlslpp::float3 worldForward;
            hlslpp::float3 worldOrigin;
            hlslpp::float4 focusPosition;
        };

        struct Caster {
            uint32_t sceneIndex;
            uint32_t instanceIndex;
            bool alphaTested;
        };

        struct ComposePipelines {
            std::unique_ptr<RenderPipeline> multiply;
            std::unique_ptr<RenderPipeline> copy;
        };

        RenderDevice *device = nullptr;
        RenderShaderFormat shaderFormat = RenderShaderFormat::UNKNOWN;
        std::unique_ptr<RenderSampler> shadowSampler;
        std::unique_ptr<RenderShader> fullScreenVertexShader;
        std::unique_ptr<RenderShader> composePixelShader;
        std::unique_ptr<RenderShader> composePixelShaderMS;
        std::unique_ptr<RenderPipelineLayout> shadowPipelineLayout;
        std::unique_ptr<RenderPipeline> shadowOpaquePipeline;
        std::unique_ptr<RenderPipeline> shadowAlphaPipeline;
        std::unique_ptr<RenderPipeline> shadowOpaquePointPipeline;
        std::unique_ptr<RenderPipeline> shadowAlphaPointPipeline;
        std::unique_ptr<RenderPipeline> shadowMergedPointPipeline;
        std::unique_ptr<RenderPipelineLayout> gbufferPipelineLayout;
        std::unique_ptr<RenderPipeline> gbufferPipelines[2][2];
        std::unique_ptr<RenderTexture> normalBuffer;
        std::unique_ptr<RenderFramebuffer> normalFramebuffer;
        std::unique_ptr<RenderTexture> normalDepth;
        uint32_t normalBufferWidth = 0;
        uint32_t normalBufferHeight = 0;
        bool foliageNormalsSupported = false;
        bool gbufferEnabled = true;
        bool mergedDraws = false;
        std::unique_ptr<RenderPipelineLayout> shadowMergedPipelineLayout;
        std::unique_ptr<RenderPipeline> shadowMergedPipeline;
        std::unique_ptr<RenderPipelineLayout> gbufferMergedPipelineLayout;
        std::unique_ptr<RenderPipeline> gbufferMergedPipelines[2];
        std::unique_ptr<LightingTriangleDrawSet> triangleDrawSet;
        BufferPair triangleDrawsBuffer;
        std::vector<uint32_t> triangleDraws;
        bool bumpEnabled = true;
        std::unique_ptr<RenderPipelineLayout> aoPipelineLayout;
        std::unique_ptr<RenderPipeline> aoPipeline;
        std::unique_ptr<RenderPipeline> aoPipelineMS;
        std::unique_ptr<RenderPipelineLayout> aoBlurPipelineLayout;
        std::unique_ptr<RenderPipeline> aoBlurPipeline;
        std::unique_ptr<RenderTexture> aoTextures[2];
        uint32_t aoTextureWidth = 0;
        uint32_t aoTextureHeight = 0;
        std::vector<std::unique_ptr<LightingAODescriptorSet>> aoSets;
        std::unique_ptr<LightingAOBlurDescriptorSet> aoBlurSets[2];
        std::unique_ptr<RenderPipelineLayout> emissivePipelineLayout;
        std::unique_ptr<RenderPipeline> emissivePipeline;
        std::unique_ptr<RenderPipeline> emissivePipelineMS;
        std::unique_ptr<RenderPipelineLayout> emissiveBlurPipelineLayout;
        std::unique_ptr<RenderPipeline> emissiveBlurPipeline;
        std::unique_ptr<RenderTexture> emissiveTextures[2];
        uint32_t emissiveTextureWidth = 0;
        uint32_t emissiveTextureHeight = 0;
        std::vector<std::unique_ptr<LightingEmissiveDescriptorSet>> emissiveSets;
        std::unique_ptr<LightingAOBlurDescriptorSet> emissiveBlurSets[2];
        uint32_t frameIndex = 0;
        std::unique_ptr<RenderPipelineLayout> composePipelineLayout;
        std::unique_ptr<RenderPipelineLayout> copyPipelineLayout;
        std::unique_ptr<RenderShader> copyPixelShader;
        std::unique_ptr<RenderShader> copyPixelShaderMS;
        std::map<std::pair<uint32_t, RenderFormat>, std::unique_ptr<RenderPipeline>> copyPipelines;
        std::vector<std::unique_ptr<LightingCopyDescriptorSet>> copySets;
        uint32_t copySetCursor = 0;
        std::unique_ptr<RenderTexture> colorCopyTexture;
        std::unique_ptr<RenderFramebuffer> colorCopyFramebuffer;
        uint32_t colorCopyWidth = 0;
        uint32_t colorCopyHeight = 0;
        RenderFormat colorCopyFormat = RenderFormat::UNKNOWN;
        std::unique_ptr<LightingSky> sky;
        std::unique_ptr<PostEffects> postEffects;
        std::map<std::pair<uint32_t, RenderFormat>, ComposePipelines> composePipelines;
        std::unique_ptr<RenderTexture> shadowMap;
        std::unique_ptr<RenderTextureView> shadowMapView;
        std::unique_ptr<RenderFramebuffer> shadowFramebuffer;
        uint32_t shadowMapSize = 0;
        bool shadowMapNeedsTransition = false;
        std::unique_ptr<RenderTexture> pointShadowMap;
        std::unique_ptr<RenderTextureView> pointShadowMapView;
        std::unique_ptr<RenderFramebuffer> pointShadowFramebuffer;
        uint32_t pointShadowFaceSize = 0;
        bool pointShadowNeedsTransition = false;
        bool pointShadowActive = false;
        bool pointShadowRendered = false;
        interop::float4x4 pointShadowMatrices[6];
        std::vector<std::unique_ptr<LightingComposeDescriptorSet>> composeSets;
        BufferPair paramsBuffer;
        std::vector<Scene> scenes;
        std::vector<interop::LightingParams> paramsVector;
        std::vector<Caster> casters;
        std::vector<Caster> sortedCasters;
        interop::float4x4 shadowMatrix;
        bool shadowMapActive = false;
        bool shadowMapRendered = false;
        uint32_t debugView = 0;

        LightingRenderer(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat);
        ~LightingRenderer();

        // Called at the start of each frame before any scene is added.
        void reset();

        // Adds a projection to light and returns its index.
        uint32_t addScene(const LightingSceneDesc &desc, RenderTarget *colorTarget, RenderTarget *depthTarget);

        // Grows the region of the framebuffer covered by a scene.
        void addSceneRect(uint32_t sceneIndex, const RenderRect &rect);

        // Adds a draw call that casts shadows.
        void addCaster(uint32_t sceneIndex, uint32_t instanceIndex, bool alphaTested);

        // Adds an opaque draw call of a scene to the normal buffer (flags are LIGHTING_GBUFFER_*).
        void addGBufferDraw(uint32_t sceneIndex, uint32_t instanceIndex, uint32_t flags);

        // Light of the glowing surfaces of a scene without a sun, before its composition (which calls it).
        void recordEmissive(RenderWorker *worker, uint32_t sceneIndex);

        // Fits the shadow map and finishes the parameters of the scenes. Adds the uploads of the frame to the list.
        void finish(RenderWorker *worker, const std::vector<InstanceDrawCall> &instanceDrawCalls, std::vector<BufferUploader::Upload> &uploads);

        // Must be called after the uploads are submitted.
        void updateDescriptorSets();

        // Draws the shadow casters into the shadow map with the same descriptors the raster shaders use.
        void recordShadowMap(RenderWorker *worker, RenderDescriptorSet *commonSet, RenderDescriptorSet *textureSet, RenderDescriptorSet *framebufferSet,
            const RenderVertexBufferView *vertexViews, const RenderInputSlot *inputSlots, uint32_t vertexViewCount, const RenderIndexBufferView *indexView,
            const std::vector<InstanceDrawCall> &instanceDrawCalls);

        // Draws the normals of the opaque surfaces of a scene. The depth target must be readable (depth read layout), and
        // the framebuffer set must be the one that binds it.
        void recordGBuffer(RenderWorker *worker, uint32_t sceneIndex, RenderDescriptorSet *commonSet, RenderDescriptorSet *textureSet, RenderDescriptorSet *framebufferSet,
            const RenderVertexBufferView *vertexViews, const RenderInputSlot *inputSlots, uint32_t vertexViewCount, const RenderIndexBufferView *indexView,
            const std::vector<InstanceDrawCall> &instanceDrawCalls);

        // Computes the ambient occlusion of a scene from its depth and normals. Must be called after recordGBuffer.
        void recordAmbientOcclusion(RenderWorker *worker, uint32_t sceneIndex);

        // Lights the color target of a scene. The depth target must be readable (depth read layout).
        void recordCompose(RenderWorker *worker, uint32_t sceneIndex);

        // Replaces the sky of the background of a scene, after it's lit. The depth target must be readable.
        void recordSky(RenderWorker *worker, uint32_t sceneIndex);

        // Applies the post effects to a scene, after its translucent surfaces are drawn. The depth target must be readable.
        void recordPostEffects(RenderWorker *worker, uint32_t sceneIndex);

        // Copies the color target of a scene (resolved if multisampled) into a texture in the shader read layout and
        // returns it. The color target is left in the color write layout.
        const RenderTexture *copyColor(RenderWorker *worker, uint32_t sceneIndex);

        bool empty() const;

    private:
        void createShadowMap(RenderWorker *worker, uint32_t size);
        void createPointShadowMap(uint32_t faceSize);
        void createNormalBuffer(RenderWorker *worker, uint32_t width, uint32_t height);
        void createAOTextures(uint32_t width, uint32_t height);
        ComposePipelines &getComposePipelines(const RenderMultisampling &multisampling, RenderFormat format);
    };

    // Computes a shadow map matrix for an orthographic sun light that covers a sphere around the camera. The light is
    // aligned to the world (given by its basis and origin in the space of the geometry) and snapped to its texels, so the
    // shadows don't shimmer when the camera moves or turns. Returns the size of a texel in world units.
    float computeStableShadowMatrix(const hlslpp::float3 &sunDirection, const hlslpp::float3 &sphereCenter, float sphereRadius, float casterDistance, uint32_t mapSize,
        const hlslpp::float3 &worldRight, const hlslpp::float3 &worldUp, const hlslpp::float3 &worldForward, const hlslpp::float3 &worldOrigin, interop::float4x4 &shadowMatrix, float &depthRange);
};
