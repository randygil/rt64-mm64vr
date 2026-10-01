//
// RT64
//

#pragma once

#include "common/rt64_plume.h"

namespace RT64 {
    using RenderHookInit = void(RenderInterface *rhi, RenderDevice *device);
    using RenderHookDraw = void(RenderCommandList *list, RenderFramebuffer *swapChainFramebuffer);
    using RenderHookDeinit = void();

    RenderHookInit *GetRenderHookInit();
    RenderHookDraw *GetRenderHookDraw();
    RenderHookDeinit *GetRenderHookDeinit();

    void SetRenderHooks(RenderHookInit *init, RenderHookDraw *draw, RenderHookDeinit *deinit);

    // Workload hooks, used by external presentation (e.g. OpenXR) to know which submitted frame is being presented.
    // Created is called on the thread that submits the display lists, once they've been turned into a workload.
    // Present is called on the present thread right before a workload's frame is drawn to the swap chain.
    using RenderHookWorkload = void(uint64_t workloadId);
    RenderHookWorkload *GetRenderHookWorkloadCreated();
    RenderHookWorkload *GetRenderHookWorkloadPresent();
    void SetRenderHookWorkload(RenderHookWorkload *created, RenderHookWorkload *present);
};
