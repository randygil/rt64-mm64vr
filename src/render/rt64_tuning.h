//
// RT64
//

#pragma once

namespace RT64 {
    // Value of a visual enhancement, which can be overridden with an environment variable of the same name or with the
    // tuning file pointed by RT64_RT_TUNING_FILE ("NAME value" per line, reloaded when it changes). Safe to call from
    // any thread.
    float enhancementValue(const char *name, float defaultValue);
};
