//
// RT64
//

#pragma once

namespace RT64 {
    struct GameConfiguration {
        float sunLightIntensity = 1.9f;
        float sunLightDistance = 5000000.0f;
        bool estimateSunLight = true;

        // Many games send the direction of their lights relative to the camera so characters always look lit, which
        // makes the shadows rotate with the camera. The sun uses a fixed direction in the world unless this is enabled.
        bool sunUsesGameLightDirection = false;
        float sunAzimuthDegrees = 35.0f;
        float sunElevationDegrees = 32.0f;
        bool rspLightAsDiffuse = true;
        float rspLightIntensity = 1.0f;
    };
};
