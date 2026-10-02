//
// RT64
//

#pragma once

#include "common/rt64_plume.h"

namespace RT64 {
    // Value of a visual enhancement, which can be overridden with an environment variable of the same name or with the
    // tuning file pointed by RT64_RT_TUNING_FILE ("NAME value" per line, reloaded when it changes). Safe to call from
    // any thread.
    float enhancementValue(const char *name, float defaultValue);

    // Strength of the visual effects added on top of the lighting (path traced or raster), from 0 to 1, as set by the
    // host with setEnhancementIntensity.
    float getEnhancementIntensity();

    // Whether the sky of outdoor scenes is replaced by a procedural one, as set by the host with setProceduralSkyEnabled.
    bool isProceduralSkyEnabled();

    // Development aid (RT64_PRINT_FRAME_TIME=2): a GPU timestamp named after the work recorded since the previous one.
    // The average of each name per frame is printed with the frame time. Only the thread that records the frame calls it.
    void gpuMarker(RenderCommandList *commandList, const char *name);

    // Used by the workload queue around the recording of a frame.
    void beginGPUMarkers(RenderCommandList *commandList, RenderQueryPool *queryPool);
    void endGPUMarkers(RenderCommandList *commandList);
    void readGPUMarkers();
    void printGPUMarkers();
};
