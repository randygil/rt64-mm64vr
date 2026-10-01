//
// RT64
//

#include "rt64_render_hooks.h"

namespace RT64 {
    static RenderHookInit *init = nullptr;
    static RenderHookDraw *draw = nullptr;
    static RenderHookDeinit *deinit = nullptr;

    RenderHookInit *GetRenderHookInit() {
        return init;
    }

    RenderHookDraw *GetRenderHookDraw() {
        return draw;
    }

    RenderHookDeinit *GetRenderHookDeinit() {
        return deinit;
    }

    void SetRenderHooks(RenderHookInit *init_, RenderHookDraw *draw_, RenderHookDeinit *deinit_) {
        init = init_;
        draw = draw_;
        deinit = deinit_;
    }

    static RenderHookWorkload *workloadCreatedHook = nullptr;
    static RenderHookWorkload *workloadPresentHook = nullptr;

    RenderHookWorkload *GetRenderHookWorkloadCreated() {
        return workloadCreatedHook;
    }

    RenderHookWorkload *GetRenderHookWorkloadPresent() {
        return workloadPresentHook;
    }

    void SetRenderHookWorkload(RenderHookWorkload *created, RenderHookWorkload *present) {
        workloadCreatedHook = created;
        workloadPresentHook = present;
    }
};
