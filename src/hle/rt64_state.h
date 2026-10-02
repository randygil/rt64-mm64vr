//
// RT64
//

#pragma once

#include <unordered_set>

#include "xxHash/xxh3.h"

#include "common/rt64_emulator_configuration.h"
#include "common/rt64_enhancement_configuration.h"
#include "gui/rt64_camera_controller.h"
#include "gui/rt64_debugger_inspector.h"
#include "preset/rt64_preset_draw_call.h"
#include "preset/rt64_preset_material.h"
#include "render/rt64_framebuffer_renderer.h"
#include "render/rt64_render_worker.h"
#include "render/rt64_raster_shader_cache.h"
#include "render/rt64_render_target_manager.h"
#include "render/rt64_rsp_processor.h"
#include "render/rt64_texture_cache.h"
#include "render/rt64_transform_processor.h"

#include "rt64_application_window.h"
#include "rt64_framebuffer.h"
#include "rt64_framebuffer_manager.h"
#include "rt64_game_configuration.h"
#include "rt64_light_manager.h"
#include "rt64_microcode.h"
#include "rt64_present_queue.h"
#include "rt64_rdp.h"
#include "rt64_rsp.h"
#include "rt64_rdp_tmem.h"
#include "rt64_workload_queue.h"

namespace RT64 {
    struct Application;
    struct Interpreter;
    
    const uint32_t RDRAMSize = 0x7FFFFF;

    struct State {
        struct External {
            Application *app;
            ApplicationWindow *appWindow;
            Interpreter *interpreter;
            RenderDevice *device;
            RenderSwapChain *swapChain;
            RenderWorker *framebufferGraphicsWorker;
            ShaderLibrary *shaderLibrary;
            BufferUploader *drawDataUploader;
            BufferUploader *transformsUploader;
            BufferUploader *tilesUploader;
            WorkloadQueue *workloadQueue;
            PresentQueue *presentQueue;
            SharedQueueResources *sharedQueueResources;
            RasterShaderCache *rasterShaderCache;
            TextureCache *textureCache;
            EmulatorConfiguration *emulatorConfig;
            EnhancementConfiguration *enhancementConfig;
            UserConfiguration *userConfig;
            ProfilingTimer *dlApiProfiler;
            ProfilingTimer *screenApiProfiler;
            UserConfiguration::GraphicsAPI createdGraphicsAPI;
#       if RT_ENABLED
            RaytracingConfiguration *rtConfig;
#       endif
        };

        uint8_t *RDRAM;
        uint32_t *MI_INTR_REG;
        void (*checkInterrupts)();
        Microcode microcode;
        std::unique_ptr<RSP> rsp;
        std::unique_ptr<RDP> rdp;
        TextureManager textureManager;
        FramebufferManager framebufferManager;
        FramebufferChangePool scratchFbChangePool;
        std::unique_ptr<FramebufferRenderer> framebufferRenderer;
        RenderTargetManager renderTargetManager;
        std::unique_ptr<RenderFramebufferManager> renderFramebufferManager;
        std::unique_ptr<RSPProcessor> rspProcessor;
        std::vector<interop::PointLight> scriptLights;
        uint64_t workloadCounter;
        std::vector<uint64_t> evictedTextureHashes;
        std::unique_ptr<RenderTarget> dummyDepthTarget;
        DrawCall drawCall;
        DrawStatus drawStatus;
        SpriteCommand activeSpriteCommand;
        bool addLightsOnFlush;
        std::vector<FramebufferOperation> drawFbOperations;
        std::vector<uint32_t> drawFbDiscards;
        std::vector<DisplayList *> returnAddressStack;
        std::vector<Framebuffer *> differentFbs;
        GameConfiguration gameConfig;

        // Rotation from a world with the Y axis pointing up to the space the game draws its geometry in, for games
        // that include the camera in their model matrices. Row major, applied as rotation * vector.
        bool worldViewRotationValid = false;
        float worldViewRotation[9] = {};
        bool worldViewTranslationValid = false;
        float worldViewTranslation[3] = {};

        // Games that draw their sky as a 2D picture before the 3D scene can ask for the sun to be left out of frames
        // that don't draw one, which are usually interiors and dungeons without a ceiling the sun could leak through.
        bool sunRequiresSkyBackground = false;

        // The sky isn't drawn when the camera looks down, so a scene is remembered as an exterior once it shows one.
        // The game can identify its scenes (e.g. the current area) with a key; without one, the sun is only left out
        // after several frames in a row without a sky.
        uint32_t sceneKey = UINT32_MAX;
        std::unordered_set<uint32_t> exteriorSceneKeys;
        uint32_t framesWithoutSky = 0;
        uint32_t displayListAddress;
        uint64_t displayListCounter;
        bool rdramCheckPending;
        uint32_t lastWorkloadIndex;
        VI lastScreenVI;
        uint64_t lastScreenHash;
        uint32_t lastScreenFactorCounter;
        VIHistory viHistory;
        PresetDrawCallLibrary drawCallLibrary;
        PresetLightsLibrary lightsLibrary;
        PresetMaterialLibrary materialLibrary;
        PresetSceneLibrary sceneLibrary;
        PresetDrawCallLibraryInspector drawCallLibraryInspector;
        PresetLightsLibraryInspector lightsLibraryInspector;
        PresetMaterialLibraryInspector materialLibraryInspector;
        PresetSceneLibraryInspector sceneLibraryInspector;
        CameraController cameraController;
        DebuggerInspector debuggerInspector;
        std::stringstream scriptLog;
        ProfilingTimer dlCpuProfiler = ProfilingTimer(120);
        ProfilingTimer screenCpuProfiler = ProfilingTimer(120);
        ProfilingTimer viChangedProfiler = ProfilingTimer(120);
        std::filesystem::path dumpingTexturesDirectory;
        bool configurationSaveQueued = false;
        uint64_t workloadId = 0;
        uint64_t presentId = 0;
        uint32_t ditherRandomSeed = 0;
        uint32_t renderFlagSampleCount = 0;
        External ext;

        struct Extended {
            hlslpp::float2 texcoordWrapPoint = { 8092.0f, 8092.0f };
            uint16_t refreshRate = UINT16_MAX;
            uint8_t renderToRAM = UINT8_MAX;
            bool vertexTestZActive = false;
            float ditherNoiseStrength = 1.0f;
            bool extendRDRAM = false;
        };

        Extended extended;

        State(uint8_t *RDRAM, uint32_t *MI_INTR_REG, void (*checkInterrupts)());
        ~State();
        void setup(const External &ext);
        void reset();
        void resetDrawCall();
        void updateDrawStatusAttribute(DrawAttribute attribute);
        bool checkDrawState();
        void loadDrawState();
        void flush();
        void submitFramebufferPair(FramebufferPair::FlushReason flushReason);
        void checkRDRAM();
        void fullSync();
        void fullSyncFramebufferPairTiles(Workload &workload, FramebufferPair &fbPair, uint32_t &loadOpCursor, uint32_t &rdpTileCursor);
        void listProcessBegin();
        void listProcessEnd();
        void updateScreen(const VI &newVI, bool fromEarlyPresent);
        void updateMultisampling();
        void inspect();
        void advanceWorkload(Workload &workload, bool paused);
        void advancePresent(Present &present, bool paused);
        void dpInterrupt();
        void spInterrupt();
        void advanceFramebufferRenderer();
        void flushFramebufferOperations(FramebufferPair &fbPair);
        bool hasFramebufferOperationsPending() const;
        void pushReturnAddress(DisplayList *dl);
        DisplayList *popReturnAddress();
        void setRefreshRate(uint16_t refreshRate);
        void setRenderToRAM(uint8_t renderToRAM);
        void setDitherNoiseStrength(float noiseStrength);
        void setExtendedRDRAM(bool isExtended);
        void setTexcoordWrapPoint(int16_t wrapU, int16_t wrapV);
        void startSpriteCommand(uint64_t replacementHash);
        void endSpriteCommand();
        uint8_t *fromRDRAM(uint32_t rdramAddress) const;
        void dumpRDRAM(const std::string &path);
        void enableExtendedGBI(uint8_t opCode);
        void disableExtendedGBI();
        void clearExtended();
        void updateRenderFlagSampleCount();
    };
};