//
// RT64
//

#include "rt64_post_effects.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <unordered_map>
#include <vector>

#include "shaders/FullScreenVS.hlsl.spirv.h"
#include "shaders/PostEffectsComposePS.hlsl.spirv.h"
#include "shaders/PostEffectsDownsamplePS.hlsl.spirv.h"
#include "shaders/PostEffectsPrefilterPS.hlsl.spirv.h"
#include "shaders/PostEffectsPrefilterPSMS.hlsl.spirv.h"
#include "shaders/PostEffectsShaftsPS.hlsl.spirv.h"
#include "shaders/PostEffectsUpsamplePS.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/FullScreenVS.hlsl.dxil.h"
#   include "shaders/PostEffectsComposePS.hlsl.dxil.h"
#   include "shaders/PostEffectsDownsamplePS.hlsl.dxil.h"
#   include "shaders/PostEffectsPrefilterPS.hlsl.dxil.h"
#   include "shaders/PostEffectsPrefilterPSMS.hlsl.dxil.h"
#   include "shaders/PostEffectsShaftsPS.hlsl.dxil.h"
#   include "shaders/PostEffectsUpsamplePS.hlsl.dxil.h"
#elif defined(__APPLE__)
#   include "shaders/FullScreenVS.hlsl.metal.h"
#   include "shaders/PostEffectsComposePS.hlsl.metal.h"
#   include "shaders/PostEffectsDownsamplePS.hlsl.metal.h"
#   include "shaders/PostEffectsPrefilterPS.hlsl.metal.h"
#   include "shaders/PostEffectsPrefilterPSMS.hlsl.metal.h"
#   include "shaders/PostEffectsShaftsPS.hlsl.metal.h"
#   include "shaders/PostEffectsUpsamplePS.hlsl.metal.h"
#endif

#include "shaders/PostEffectsParams.hlsli"

#include "rt64_lighting.h"
#include "rt64_shader_library.h"
#include "rt64_tuning.h"

#ifdef _WIN32
#   define POST_EFFECTS_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? NAME##BlobDXIL : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::DXIL) ? sizeof(NAME##BlobDXIL) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#elif defined(__APPLE__)
#   define POST_EFFECTS_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? NAME##BlobMSL : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::METAL) ? sizeof(NAME##BlobMSL) : (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#else
#   define POST_EFFECTS_SHADER_INPUTS(NAME, ENTRY_NAME, SHADER_FORMAT)\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? NAME##BlobSPIRV : nullptr,\
        (SHADER_FORMAT == RenderShaderFormat::SPIRV) ? sizeof(NAME##BlobSPIRV) : 0,\
        ENTRY_NAME,\
        SHADER_FORMAT
#endif

namespace RT64 {
    struct PostEffectsPrefilterDescriptorSet : RenderDescriptorSetBase {
        uint32_t gSceneColor;
        uint32_t gDepth;
        uint32_t gLinearSampler;

        PostEffectsPrefilterDescriptorSet(const RenderSampler *linearSampler, RenderDevice *device = nullptr) {
            builder.begin();
            gSceneColor = builder.addTexture(1);
            gDepth = builder.addTexture(2);
            gLinearSampler = builder.addImmutableSampler(3, linearSampler);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct PostEffectsInputDescriptorSet : RenderDescriptorSetBase {
        uint32_t gInput;
        uint32_t gLinearSampler;

        PostEffectsInputDescriptorSet(const RenderSampler *linearSampler, RenderDevice *device = nullptr) {
            builder.begin();
            gInput = builder.addTexture(1);
            gLinearSampler = builder.addImmutableSampler(2, linearSampler);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct PostEffectsComposeDescriptorSet : RenderDescriptorSetBase {
        uint32_t gSceneColor;
        uint32_t gBloom;
        uint32_t gShafts;
        uint32_t gLinearSampler;

        PostEffectsComposeDescriptorSet(const RenderSampler *linearSampler, RenderDevice *device = nullptr) {
            builder.begin();
            gSceneColor = builder.addTexture(1);
            gBloom = builder.addTexture(2);
            gShafts = builder.addTexture(3);
            gLinearSampler = builder.addImmutableSampler(4, linearSampler);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // Cost of the effects for each quality level of the enhanced lighting (low, medium, high, ultra).
    static const uint32_t QualityBloomLevels[4] = { 4, 5, 6, 6 };
    static const uint32_t QualityShaftDivisors[4] = { 4, 4, 2, 2 };
    static const uint32_t QualityShaftPasses[4] = { 1, 2, 2, 2 };
    static const uint32_t QualityShaftSamples[4] = { 16, 12, 16, 24 };

    // Values of the effects, read on every frame so they can be tuned live.
    struct PostEffectsSettings {
        bool enabled = false;
        uint32_t quality = 0;
        float bloomStrength = 0.0f;
        float bloomThreshold = 0.0f;
        float bloomKnee = 0.0f;
        float bloomScatter = 0.0f;
        float bloomRadius = 0.0f;
        uint32_t bloomLevels = 1;
        float shaftsStrength = 0.0f;
        float shaftsDensity = 0.0f;
        float shaftsFalloff = 0.0f;
        float shaftsThreshold = 0.0f;
        float shaftsRadius = 0.0f;
        float shaftsEdgeFade = 0.0f;
        float sharpen = 0.0f;
        float contrast = 0.0f;
        float vibrance = 0.0f;
        float saturation = 0.0f;
        float temperature = 0.0f;
        float vignette = 0.0f;
        float dither = 0.0f;
    };

    static PostEffectsSettings readPostEffectsSettings() {
        PostEffectsSettings s;
        const int qualityOverride = int(enhancementValue("RT64_POST_QUALITY", -1.0f));
        s.quality = uint32_t(std::clamp((qualityOverride >= 0) ? qualityOverride : getRasterLightingQuality(), 0, 3));
        s.bloomStrength = std::max(enhancementValue("RT64_POST_BLOOM_STRENGTH", 0.3f), 0.0f);
        s.bloomThreshold = enhancementValue("RT64_POST_BLOOM_THRESHOLD", 0.7f);
        s.bloomKnee = std::max(enhancementValue("RT64_POST_BLOOM_KNEE", 0.2f), 0.0f);
        s.bloomScatter = std::clamp(enhancementValue("RT64_POST_BLOOM_SCATTER", 0.65f), 0.0f, 1.0f);
        s.bloomRadius = std::max(enhancementValue("RT64_POST_BLOOM_RADIUS", 1.0f), 0.0f);
        const int bloomLevels = int(enhancementValue("RT64_POST_BLOOM_LEVELS", 0.0f));
        s.bloomLevels = uint32_t(std::clamp((bloomLevels > 0) ? bloomLevels : int(QualityBloomLevels[s.quality]), 1, 8));
        s.shaftsStrength = std::max(enhancementValue("RT64_POST_SHAFTS_STRENGTH", 0.35f), 0.0f);
        s.shaftsDensity = std::clamp(enhancementValue("RT64_POST_SHAFTS_DENSITY", 0.9f), 0.05f, 1.0f);
        s.shaftsFalloff = std::max(enhancementValue("RT64_POST_SHAFTS_FALLOFF", 1.0f), 0.0f);
        s.shaftsThreshold = std::clamp(enhancementValue("RT64_POST_SHAFTS_THRESHOLD", 0.35f), 0.0f, 0.99f);
        s.shaftsRadius = std::max(enhancementValue("RT64_POST_SHAFTS_RADIUS", 0.6f), 0.01f);
        s.shaftsEdgeFade = std::max(enhancementValue("RT64_POST_SHAFTS_EDGE_FADE", 0.4f), 0.0f);
        s.sharpen = std::clamp(enhancementValue("RT64_POST_SHARPEN", 0.35f), 0.0f, 1.0f);
        s.contrast = enhancementValue("RT64_POST_CONTRAST", 0.1f);
        s.vibrance = enhancementValue("RT64_POST_VIBRANCE", 0.15f);
        s.saturation = enhancementValue("RT64_POST_SATURATION", 0.0f);
        s.temperature = enhancementValue("RT64_POST_TEMPERATURE", 0.0f);
        s.vignette = std::max(enhancementValue("RT64_POST_VIGNETTE", 0.1f), 0.0f);
        s.dither = std::max(enhancementValue("RT64_POST_DITHER", 1.0f), 0.0f);

        // The host's choice of how strong the effects are (shared with the path tracer's effects).
        const float intensity = getEnhancementIntensity();
        s.bloomStrength *= intensity;
        s.shaftsStrength *= intensity;
        s.sharpen *= intensity;
        s.contrast *= intensity;
        s.vibrance *= intensity;
        s.saturation *= intensity;
        s.temperature *= intensity;
        s.vignette *= intensity;

        // Dithering alone doesn't justify the pass.
        const bool anyEffect = (s.bloomStrength > 0.0f) || (s.shaftsStrength > 0.0f) || (s.sharpen > 0.0f) || (s.contrast != 0.0f) ||
            (s.vibrance != 0.0f) || (s.saturation != 0.0f) || (s.temperature != 0.0f) || (s.vignette > 0.0f);

        s.enabled = anyEffect && (enhancementValue("RT64_POST_ENABLE", 1.0f) > 0.0f);
        return s;
    }

    static uint32_t halfSize(uint32_t size) {
        return std::max((size + 1) / 2, 1U);
    }

    static uint32_t divideRoundUp(uint32_t value, uint32_t divisor) {
        return (value + divisor - 1) / divisor;
    }

    // Size of one step of the color target, so the dithering noise covers exactly one step.
    static float colorFormatStep(RenderFormat format) {
        switch (format) {
        case RenderFormat::R16G16B16A16_UNORM:
            return 1.0f / 65535.0f;
        default:
            return 1.0f / 255.0f;
        }
    }

    // Finds the sun in the pixels of the target and how visible its light shafts are: they fade out as the sun leaves the
    // region of the scene. Returns false if there's no sun or it's behind the camera.
    static bool placeSun(const interop::LightingParams &lighting, const RenderRect &rect, float edgeFade, float &sunX, float &sunY, float &visibility) {
        if (lighting.sunDirection.w <= 0.0f) {
            return false;
        }

        // The sun is infinitely far away, so its direction projects to the point all the rays towards it converge to.
        const hlslpp::float4x4 viewProj = lighting.viewProj;
        const hlslpp::float4 clip = hlslpp::mul(hlslpp::float4(lighting.sunDirection.x, lighting.sunDirection.y, lighting.sunDirection.z, 0.0f), viewProj);
        const float clipW = float(clip.w);
        if ((clipW <= 1e-6f) || (fabsf(lighting.pixelToClip.x) < 1e-12f) || (fabsf(lighting.pixelToClip.y) < 1e-12f)) {
            return false;
        }

        // Inverse of the mapping from pixels to the clip space of the projection.
        sunX = (float(clip.x) / clipW - lighting.pixelToClip.z) / lighting.pixelToClip.x;
        sunY = (float(clip.y) / clipW - lighting.pixelToClip.w) / lighting.pixelToClip.y;

        const float outsideX = std::max(std::max(float(rect.left) - sunX, sunX - float(rect.right)), 0.0f);
        const float outsideY = std::max(std::max(float(rect.top) - sunY, sunY - float(rect.bottom)), 0.0f);
        const float outside = sqrtf(outsideX * outsideX + outsideY * outsideY) / float(rect.bottom - rect.top);
        const float fade = (edgeFade > 0.0f) ? std::clamp(1.0f - outside / edgeFade, 0.0f, 1.0f) : ((outside > 0.0f) ? 0.0f : 1.0f);
        visibility = fade * fade * (3.0f - 2.0f * fade);
        return true;
    }

    // PostEffects::Impl

    struct PostEffects::Impl {
        // Texture drawn by a pass and the region of it in use, which can be smaller.
        struct PassTarget {
            std::unique_ptr<RenderTexture> texture;
            std::unique_ptr<RenderFramebuffer> framebuffer;
            uint32_t textureWidth = 0;
            uint32_t textureHeight = 0;
            uint32_t width = 0;
            uint32_t height = 0;
        };

        // Resources of a scene. Every scene has its own, so they can be resized and their descriptors updated while it's
        // recorded: the last commands that used them belong to a previous frame, which has finished by then.
        struct Scene {
            std::vector<PassTarget> bloomLevels;
            PassTarget shaftTargets[2];
            std::unique_ptr<PostEffectsPrefilterDescriptorSet> prefilterSet;
            std::vector<std::unique_ptr<PostEffectsInputDescriptorSet>> downsampleSets;
            std::vector<std::unique_ptr<PostEffectsInputDescriptorSet>> upsampleSets;
            std::unique_ptr<PostEffectsInputDescriptorSet> shaftSets[2];
            std::unique_ptr<PostEffectsComposeDescriptorSet> composeSet;
            uint32_t capacityWidth = 0;
            uint32_t capacityHeight = 0;
            uint32_t shaftDivisor = 0;
            uint64_t lastFrame = 0;
        };

        RenderDevice *device = nullptr;
        const RenderSampler *linearSampler = nullptr;
        std::unique_ptr<RenderShader> fullScreenVertexShader;
        std::unique_ptr<RenderShader> composePixelShader;
        std::unique_ptr<RenderPipelineLayout> prefilterLayout;
        std::unique_ptr<RenderPipelineLayout> downsampleLayout;
        std::unique_ptr<RenderPipelineLayout> upsampleLayout;
        std::unique_ptr<RenderPipelineLayout> shaftsLayout;
        std::unique_ptr<RenderPipelineLayout> composeLayout;
        std::unique_ptr<RenderPipeline> prefilterPipeline;
        std::unique_ptr<RenderPipeline> prefilterPipelineMS;
        std::unique_ptr<RenderPipeline> downsamplePipeline;
        std::unique_ptr<RenderPipeline> upsamplePipeline;
        std::unique_ptr<RenderPipeline> shaftsPipeline;
        std::map<std::pair<uint32_t, RenderFormat>, std::unique_ptr<RenderPipeline>> composePipelines;

        // Scenes are told apart by the address of their lighting parameters, which is unique among the scenes of a frame.
        std::unordered_map<const interop::LightingParams *, std::unique_ptr<Scene>> scenes;
        uint64_t frameIndex = 1;

        Impl(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat) {
            this->device = device;
            linearSampler = shaderLibrary->samplerLibrary.linear.clampClamp.get();
            fullScreenVertexShader = device->createShader(POST_EFFECTS_SHADER_INPUTS(FullScreenVS, "VSMain", shaderFormat));
            composePixelShader = device->createShader(POST_EFFECTS_SHADER_INPUTS(PostEffectsComposePS, "PSMain", shaderFormat));
            std::unique_ptr<RenderShader> prefilterPixelShader = device->createShader(POST_EFFECTS_SHADER_INPUTS(PostEffectsPrefilterPS, "PSMain", shaderFormat));
            std::unique_ptr<RenderShader> prefilterPixelShaderMS = device->createShader(POST_EFFECTS_SHADER_INPUTS(PostEffectsPrefilterPSMS, "PSMain", shaderFormat));
            std::unique_ptr<RenderShader> downsamplePixelShader = device->createShader(POST_EFFECTS_SHADER_INPUTS(PostEffectsDownsamplePS, "PSMain", shaderFormat));
            std::unique_ptr<RenderShader> upsamplePixelShader = device->createShader(POST_EFFECTS_SHADER_INPUTS(PostEffectsUpsamplePS, "PSMain", shaderFormat));
            std::unique_ptr<RenderShader> shaftsPixelShader = device->createShader(POST_EFFECTS_SHADER_INPUTS(PostEffectsShaftsPS, "PSMain", shaderFormat));

            auto createLayout = [&](const RenderDescriptorSetBase &descriptorSet, uint32_t pushConstantSize) {
                RenderPipelineLayoutBuilder layoutBuilder;
                layoutBuilder.begin();
                layoutBuilder.addPushConstant(0, 0, pushConstantSize, RenderShaderStageFlag::PIXEL);
                layoutBuilder.addDescriptorSet(descriptorSet);
                layoutBuilder.end();
                return layoutBuilder.create(device);
            };

            prefilterLayout = createLayout(PostEffectsPrefilterDescriptorSet(linearSampler), sizeof(interop::PostEffectsPrefilterCB));
            downsampleLayout = createLayout(PostEffectsInputDescriptorSet(linearSampler), sizeof(interop::PostEffectsDownsampleCB));
            upsampleLayout = createLayout(PostEffectsInputDescriptorSet(linearSampler), sizeof(interop::PostEffectsUpsampleCB));
            shaftsLayout = createLayout(PostEffectsInputDescriptorSet(linearSampler), sizeof(interop::PostEffectsShaftsCB));
            composeLayout = createLayout(PostEffectsComposeDescriptorSet(linearSampler), sizeof(interop::PostEffectsComposeCB));

            // The passes between the scene and the composition are fragment passes, which work the same on every backend
            // and on tiled mobile GPUs, and don't need the scene's textures to be readable from compute shaders.
            auto createPassPipeline = [&](const RenderPipelineLayout *pipelineLayout, const RenderShader *pixelShader, RenderFormat format, const RenderBlendDesc &blendDesc) {
                RenderGraphicsPipelineDesc pipelineDesc;
                pipelineDesc.pipelineLayout = pipelineLayout;
                pipelineDesc.vertexShader = fullScreenVertexShader.get();
                pipelineDesc.pixelShader = pixelShader;
                pipelineDesc.renderTargetFormat[0] = format;
                pipelineDesc.renderTargetBlend[0] = blendDesc;
                pipelineDesc.renderTargetCount = 1;
                pipelineDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
                pipelineDesc.cullMode = RenderCullMode::NONE;
                return device->createGraphicsPipeline(pipelineDesc);
            };

            // The upsampling is blended over the level below with the alpha the shader outputs, and keeps the shaft mask
            // stored in the alpha channel of the first level.
            RenderBlendDesc upsampleBlend = RenderBlendDesc::AlphaBlend();
            upsampleBlend.renderTargetWriteMask = colorWriteMask();

            prefilterPipeline = createPassPipeline(prefilterLayout.get(), prefilterPixelShader.get(), RenderFormat::R16G16B16A16_FLOAT, RenderBlendDesc::Copy());
            prefilterPipelineMS = createPassPipeline(prefilterLayout.get(), prefilterPixelShaderMS.get(), RenderFormat::R16G16B16A16_FLOAT, RenderBlendDesc::Copy());
            downsamplePipeline = createPassPipeline(downsampleLayout.get(), downsamplePixelShader.get(), RenderFormat::R16G16B16A16_FLOAT, RenderBlendDesc::Copy());
            upsamplePipeline = createPassPipeline(upsampleLayout.get(), upsamplePixelShader.get(), RenderFormat::R16G16B16A16_FLOAT, upsampleBlend);
            shaftsPipeline = createPassPipeline(shaftsLayout.get(), shaftsPixelShader.get(), RenderFormat::R16_FLOAT, RenderBlendDesc::Copy());
        }

        static uint8_t colorWriteMask() {
            return uint8_t(RenderColorWriteEnable::RED) | uint8_t(RenderColorWriteEnable::GREEN) | uint8_t(RenderColorWriteEnable::BLUE);
        }

        Scene &getScene(const interop::LightingParams *key) {
            std::unique_ptr<Scene> &scene = scenes[key];
            if (scene == nullptr) {
                scene = std::make_unique<Scene>();
            }
            else if (scene->lastFrame == frameIndex) {
                // A scene is recorded once per frame, so seeing it again means a new frame started and the frames
                // recorded before it have finished. The resources of scenes missing from the last frame are released.
                frameIndex++;
                for (auto it = scenes.begin(); it != scenes.end();) {
                    if ((it->second->lastFrame + 2) <= frameIndex) {
                        it = scenes.erase(it);
                    }
                    else {
                        it++;
                    }
                }
            }

            scene->lastFrame = frameIndex;
            return *scene;
        }

        void createTarget(PassTarget &target, uint32_t width, uint32_t height, RenderFormat format, const char *name) {
            target.framebuffer.reset();
            target.texture = device->createTexture(RenderTextureDesc::ColorTarget(width, height, format));
            target.texture->setName(name);
            const RenderTexture *colorAttachment = target.texture.get();
            target.framebuffer = device->createFramebuffer(RenderFramebufferDesc(&colorAttachment, 1));
            target.textureWidth = width;
            target.textureHeight = height;
        }

        void setupScene(Scene &scene, uint32_t width, uint32_t height, uint32_t levelCount, uint32_t shaftDivisor) {
            if (scene.prefilterSet == nullptr) {
                scene.prefilterSet = std::make_unique<PostEffectsPrefilterDescriptorSet>(linearSampler, device);
                scene.composeSet = std::make_unique<PostEffectsComposeDescriptorSet>(linearSampler, device);
            }

            const bool fits = (width <= scene.capacityWidth) && (height <= scene.capacityHeight);
            if (fits && (scene.bloomLevels.size() == levelCount) && (scene.shaftDivisor == shaftDivisor)) {
                return;
            }

            // Rounded up so small changes of the region don't recreate the textures.
            const uint32_t Granularity = 64;
            scene.capacityWidth = divideRoundUp(std::max(width, scene.capacityWidth), Granularity) * Granularity;
            scene.capacityHeight = divideRoundUp(std::max(height, scene.capacityHeight), Granularity) * Granularity;
            scene.shaftDivisor = shaftDivisor;

            scene.downsampleSets.clear();
            scene.upsampleSets.clear();
            scene.bloomLevels.clear();
            scene.bloomLevels.resize(levelCount);
            uint32_t levelWidth = scene.capacityWidth;
            uint32_t levelHeight = scene.capacityHeight;
            for (PassTarget &level : scene.bloomLevels) {
                levelWidth = halfSize(levelWidth);
                levelHeight = halfSize(levelHeight);
                createTarget(level, levelWidth, levelHeight, RenderFormat::R16G16B16A16_FLOAT, "Post Effects Bloom");
            }

            for (PassTarget &target : scene.shaftTargets) {
                createTarget(target, divideRoundUp(scene.capacityWidth, shaftDivisor), divideRoundUp(scene.capacityHeight, shaftDivisor), RenderFormat::R16_FLOAT, "Post Effects Shafts");
            }

            // The descriptors between the passes only change along with the textures.
            auto createInputSet = [&](const PassTarget &input) {
                std::unique_ptr<PostEffectsInputDescriptorSet> descriptorSet = std::make_unique<PostEffectsInputDescriptorSet>(linearSampler, device);
                descriptorSet->setTexture(descriptorSet->gInput, input.texture.get(), RenderTextureLayout::SHADER_READ);
                return descriptorSet;
            };

            for (uint32_t i = 1; i < levelCount; i++) {
                scene.downsampleSets.emplace_back(createInputSet(scene.bloomLevels[i - 1]));
                scene.upsampleSets.emplace_back(createInputSet(scene.bloomLevels[i]));
            }

            scene.shaftSets[0] = createInputSet(scene.bloomLevels[0]);
            scene.shaftSets[1] = createInputSet(scene.shaftTargets[0]);
        }

        RenderPipeline *getComposePipeline(const RenderMultisampling &multisampling, RenderFormat format) {
            std::unique_ptr<RenderPipeline> &pipeline = composePipelines[{ multisampling.sampleCount, format }];
            if (pipeline == nullptr) {
                RenderGraphicsPipelineDesc pipelineDesc;
                pipelineDesc.pipelineLayout = composeLayout.get();
                pipelineDesc.vertexShader = fullScreenVertexShader.get();
                pipelineDesc.pixelShader = composePixelShader.get();
                pipelineDesc.renderTargetFormat[0] = format;
                pipelineDesc.renderTargetCount = 1;
                pipelineDesc.multisampling = multisampling;
                pipelineDesc.primitiveTopology = RenderPrimitiveTopology::TRIANGLE_LIST;
                pipelineDesc.cullMode = RenderCullMode::NONE;

                // The alpha channel holds the coverage of the RDP and must be kept.
                pipelineDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
                pipelineDesc.renderTargetBlend[0].renderTargetWriteMask = colorWriteMask();
                pipeline = device->createGraphicsPipeline(pipelineDesc);
            }

            return pipeline.get();
        }

        void drawPass(RenderCommandList *commandList, const PassTarget &target, const RenderPipeline *pipeline, const RenderPipelineLayout *pipelineLayout,
            RenderDescriptorSet *descriptorSet, const void *pushConstants)
        {
            commandList->setFramebuffer(target.framebuffer.get());
            commandList->setViewports(RenderViewport(0.0f, 0.0f, float(target.width), float(target.height)));
            commandList->setScissors(RenderRect(0, 0, int32_t(target.width), int32_t(target.height)));
            commandList->setPipeline(pipeline);
            commandList->setGraphicsPipelineLayout(pipelineLayout);
            commandList->setGraphicsDescriptorSet(descriptorSet, 0);
            commandList->setGraphicsPushConstants(0, pushConstants);
            commandList->drawInstanced(3, 1, 0, 0);
        }

        void record(RenderWorker *worker, const PostEffectsSceneDesc &desc) {
            assert(worker != nullptr);
            assert(desc.colorTarget != nullptr);
            assert(desc.sceneColor != nullptr);

            const PostEffectsSettings settings = readPostEffectsSettings();
            if (!settings.enabled) {
                return;
            }

            RenderTarget *colorTarget = desc.colorTarget;
            RenderRect rect = desc.rect;
            rect.left = std::max(rect.left, 0);
            rect.top = std::max(rect.top, 0);
            rect.right = std::min(rect.right, int32_t(colorTarget->width));
            rect.bottom = std::min(rect.bottom, int32_t(colorTarget->height));
            if (rect.isEmpty()) {
                return;
            }

            const uint32_t width = uint32_t(rect.right - rect.left);
            const uint32_t height = uint32_t(rect.bottom - rect.top);
            const uint32_t shaftDivisor = QualityShaftDivisors[settings.quality];
            const uint32_t shaftPasses = QualityShaftPasses[settings.quality];
            const uint32_t shaftSamples = QualityShaftSamples[settings.quality];
            const bool wideFilter = (settings.quality > 0);
            Scene &scene = getScene(desc.lighting);
            setupScene(scene, width, height, settings.bloomLevels, shaftDivisor);

            uint32_t levelWidth = width;
            uint32_t levelHeight = height;
            for (PassTarget &level : scene.bloomLevels) {
                levelWidth = halfSize(levelWidth);
                levelHeight = halfSize(levelHeight);
                level.width = levelWidth;
                level.height = levelHeight;
            }

            for (PassTarget &target : scene.shaftTargets) {
                target.width = divideRoundUp(width, shaftDivisor);
                target.height = divideRoundUp(height, shaftDivisor);
            }

            // The first pass reads the depth, so both effects need it.
            float sunX = 0.0f;
            float sunY = 0.0f;
            float sunVisibility = 0.0f;
            const bool hasDepth = (desc.depthTarget != nullptr) && (desc.lighting != nullptr);
            const bool sunPlaced = hasDepth && placeSun(*desc.lighting, rect, settings.shaftsEdgeFade, sunX, sunY, sunVisibility);
            const bool bloomActive = hasDepth && (settings.bloomStrength > 0.0f);
            const bool shaftsActive = sunPlaced && (settings.shaftsStrength > 0.0f) && (sunVisibility > 0.0f);
            PassTarget &firstLevel = scene.bloomLevels[0];
            const PassTarget &shaftOutput = scene.shaftTargets[shaftPasses - 1];
            RenderCommandList *commandList = worker->commandList.get();
            commandList->setVertexBuffers(0, nullptr, 0, nullptr);

            if (bloomActive || shaftsActive) {
                PostEffectsPrefilterDescriptorSet *prefilterSet = scene.prefilterSet.get();
                prefilterSet->setTexture(prefilterSet->gSceneColor, desc.sceneColor, RenderTextureLayout::SHADER_READ);
                prefilterSet->setTexture(prefilterSet->gDepth, desc.depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, desc.depthTarget->textureView.get());

                interop::PostEffectsPrefilterCB prefilterCB = {};
                prefilterCB.sourceRect = interop::int4(rect.left, rect.top, rect.right - 1, rect.bottom - 1);
                prefilterCB.sunPosition = interop::float2(sunX, sunY);
                prefilterCB.sunInvRadius = 1.0f / (settings.shaftsRadius * float(height));
                prefilterCB.backgroundDepth = desc.lighting->depthToClip.z;
                prefilterCB.bloomThreshold = settings.bloomThreshold;
                prefilterCB.bloomKnee = settings.bloomKnee;
                prefilterCB.shaftThreshold = settings.shaftsThreshold;
                prefilterCB.flags = (wideFilter ? POST_EFFECTS_FLAG_WIDE_FILTER : 0) | (shaftsActive ? POST_EFFECTS_FLAG_SHAFT_MASK : 0);

                const bool depthMultisampled = (desc.depthTarget->multisampling.sampleCount > 1);
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(firstLevel.texture.get(), RenderTextureLayout::COLOR_WRITE));
                drawPass(commandList, firstLevel, depthMultisampled ? prefilterPipelineMS.get() : prefilterPipeline.get(), prefilterLayout.get(), prefilterSet->get(), &prefilterCB);
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(firstLevel.texture.get(), RenderTextureLayout::SHADER_READ));
            }

            if (shaftsActive) {
                const float rectLeft = float(rect.left);
                const float rectTop = float(rect.top);
                interop::PostEffectsShaftsCB shaftsCB = {};
                shaftsCB.channelMask = interop::float4(0.0f, 0.0f, 0.0f, 1.0f);
                shaftsCB.inputInvSize = interop::float2(1.0f / float(firstLevel.textureWidth), 1.0f / float(firstLevel.textureHeight));
                shaftsCB.inputSize = interop::float2(float(firstLevel.width), float(firstLevel.height));
                shaftsCB.sunPosition = interop::float2((sunX - rectLeft) / 2.0f, (sunY - rectTop) / 2.0f);
                shaftsCB.inputScale = float(shaftDivisor) / 2.0f;
                shaftsCB.density = settings.shaftsDensity;
                shaftsCB.decay = expf(-settings.shaftsFalloff / float(shaftSamples));
                shaftsCB.sampleCount = shaftSamples;
                shaftsCB.flags = (shaftPasses == 1) ? POST_EFFECTS_FLAG_JITTER : 0;

                const PassTarget &firstShafts = scene.shaftTargets[0];
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(firstShafts.texture.get(), RenderTextureLayout::COLOR_WRITE));
                drawPass(commandList, firstShafts, shaftsPipeline.get(), shaftsLayout.get(), scene.shaftSets[0]->get(), &shaftsCB);

                if (shaftPasses > 1) {
                    // The second pass averages over the length of one step of the first one, towards the sun as well.
                    const PassTarget &secondShafts = scene.shaftTargets[1];
                    shaftsCB.channelMask = interop::float4(1.0f, 0.0f, 0.0f, 0.0f);
                    shaftsCB.inputInvSize = interop::float2(1.0f / float(firstShafts.textureWidth), 1.0f / float(firstShafts.textureHeight));
                    shaftsCB.inputSize = interop::float2(float(firstShafts.width), float(firstShafts.height));
                    shaftsCB.sunPosition = interop::float2((sunX - rectLeft) / float(shaftDivisor), (sunY - rectTop) / float(shaftDivisor));
                    shaftsCB.inputScale = 1.0f;
                    shaftsCB.density = settings.shaftsDensity / float(shaftSamples);
                    shaftsCB.decay = 1.0f;
                    shaftsCB.flags = 0;
                    commandList->barriers(RenderBarrierStage::GRAPHICS, {
                        RenderTextureBarrier(firstShafts.texture.get(), RenderTextureLayout::SHADER_READ),
                        RenderTextureBarrier(secondShafts.texture.get(), RenderTextureLayout::COLOR_WRITE)
                    });

                    drawPass(commandList, secondShafts, shaftsPipeline.get(), shaftsLayout.get(), scene.shaftSets[1]->get(), &shaftsCB);
                }
            }

            if (bloomActive) {
                const uint32_t levelCount = uint32_t(scene.bloomLevels.size());
                interop::PostEffectsDownsampleCB downsampleCB = {};
                downsampleCB.flags = wideFilter ? POST_EFFECTS_FLAG_WIDE_FILTER : 0;
                for (uint32_t i = 1; i < levelCount; i++) {
                    const PassTarget &input = scene.bloomLevels[i - 1];
                    const PassTarget &output = scene.bloomLevels[i];
                    downsampleCB.inputInvSize = interop::float2(1.0f / float(input.textureWidth), 1.0f / float(input.textureHeight));
                    downsampleCB.inputSize = interop::float2(float(input.width), float(input.height));
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(output.texture.get(), RenderTextureLayout::COLOR_WRITE));
                    drawPass(commandList, output, downsamplePipeline.get(), downsampleLayout.get(), scene.downsampleSets[i - 1]->get(), &downsampleCB);
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(output.texture.get(), RenderTextureLayout::SHADER_READ));
                }

                interop::PostEffectsUpsampleCB upsampleCB = {};
                upsampleCB.radius = settings.bloomRadius;
                upsampleCB.scatter = settings.bloomScatter;
                upsampleCB.flags = wideFilter ? POST_EFFECTS_FLAG_WIDE_FILTER : 0;
                for (uint32_t i = levelCount - 1; i > 0; i--) {
                    const PassTarget &input = scene.bloomLevels[i];
                    const PassTarget &output = scene.bloomLevels[i - 1];
                    upsampleCB.inputInvSize = interop::float2(1.0f / float(input.textureWidth), 1.0f / float(input.textureHeight));
                    upsampleCB.inputSize = interop::float2(float(input.width), float(input.height));
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(output.texture.get(), RenderTextureLayout::COLOR_WRITE));
                    drawPass(commandList, output, upsamplePipeline.get(), upsampleLayout.get(), scene.upsampleSets[i - 1]->get(), &upsampleCB);
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(output.texture.get(), RenderTextureLayout::SHADER_READ));
                }
            }

            // Everything the composition binds must be readable, including the textures of the effects that are off.
            commandList->barriers(RenderBarrierStage::GRAPHICS, {
                RenderTextureBarrier(firstLevel.texture.get(), RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(shaftOutput.texture.get(), RenderTextureLayout::SHADER_READ)
            });

            interop::PostEffectsComposeCB composeCB = {};
            composeCB.rect = interop::int4(rect.left, rect.top, rect.right - 1, rect.bottom - 1);
            composeCB.bloomInvSize = interop::float2(1.0f / float(firstLevel.textureWidth), 1.0f / float(firstLevel.textureHeight));
            composeCB.bloomSize = interop::float2(float(firstLevel.width), float(firstLevel.height));
            composeCB.shaftsInvSize = interop::float2(1.0f / float(shaftOutput.textureWidth), 1.0f / float(shaftOutput.textureHeight));
            composeCB.shaftsSize = interop::float2(float(shaftOutput.width), float(shaftOutput.height));
            composeCB.shaftsColor = interop::float3(0.0f, 0.0f, 0.0f);
            if (shaftsActive) {
                // Tinted by the color of the sun but not by its intensity, which the strength already sets.
                const interop::float4 &sunColor = desc.lighting->sunColor;
                const float maxComponent = std::max(std::max(sunColor.x, sunColor.y), sunColor.z);
                const float scale = settings.shaftsStrength * sunVisibility;
                if (maxComponent > 1e-4f) {
                    composeCB.shaftsColor = interop::float3(sunColor.x * scale / maxComponent, sunColor.y * scale / maxComponent, sunColor.z * scale / maxComponent);
                }
                else {
                    composeCB.shaftsColor = interop::float3(scale, scale, scale);
                }
            }

            composeCB.bloomStrength = bloomActive ? settings.bloomStrength : 0.0f;
            composeCB.shaftsScale = 1.0f / float(shaftDivisor);
            composeCB.sharpenPeak = (settings.sharpen > 0.0f) ? (-1.0f / (8.0f + (5.0f - 8.0f) * settings.sharpen)) : 0.0f;
            composeCB.contrast = settings.contrast;
            composeCB.vibrance = settings.vibrance;
            composeCB.saturation = settings.saturation;
            composeCB.temperature = settings.temperature;
            composeCB.vignette = settings.vignette;
            composeCB.ditherAmplitude = settings.dither * colorFormatStep(colorTarget->format);
            composeCB.flags = wideFilter ? POST_EFFECTS_FLAG_SHARPEN_DIAGONALS : 0;

            PostEffectsComposeDescriptorSet *composeSet = scene.composeSet.get();
            composeSet->setTexture(composeSet->gSceneColor, desc.sceneColor, RenderTextureLayout::SHADER_READ);
            composeSet->setTexture(composeSet->gBloom, firstLevel.texture.get(), RenderTextureLayout::SHADER_READ);
            composeSet->setTexture(composeSet->gShafts, shaftOutput.texture.get(), RenderTextureLayout::SHADER_READ);

            colorTarget->setupColorFramebuffer(worker);
            commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
            commandList->setViewports(RenderViewport(0.0f, 0.0f, float(colorTarget->width), float(colorTarget->height)));
            commandList->setScissors(rect);
            commandList->setPipeline(getComposePipeline(colorTarget->multisampling, colorTarget->format));
            commandList->setGraphicsPipelineLayout(composeLayout.get());
            commandList->setGraphicsDescriptorSet(composeSet->get(), 0);
            commandList->setGraphicsPushConstants(0, &composeCB);
            commandList->drawInstanced(3, 1, 0, 0);
        }
    };

    // PostEffects

    PostEffects::PostEffects(RenderDevice *device, const ShaderLibrary *shaderLibrary, RenderShaderFormat shaderFormat) {
        assert(device != nullptr);
        assert(shaderLibrary != nullptr);

        impl = std::make_unique<Impl>(device, shaderLibrary, shaderFormat);
    }

    PostEffects::~PostEffects() { }

    bool PostEffects::enabled() const {
        return readPostEffectsSettings().enabled;
    }

    void PostEffects::record(RenderWorker *worker, const PostEffectsSceneDesc &desc) {
        impl->record(worker, desc);
    }
};
