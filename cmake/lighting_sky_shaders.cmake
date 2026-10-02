# Shaders of the procedural sky of the enhanced lighting (render/rt64_lighting_sky.cpp). Included from CMakeLists.txt after the shader build functions.
build_pixel_shader(rt64 "src/shaders/LightingSkyLutPS.hlsl")
build_pixel_shader(rt64 "src/shaders/LightingSkyPS.hlsl")
build_pixel_shader(rt64 "src/shaders/LightingSkyPS.hlsl" "src/shaders/LightingSkyPSMS.hlsl" "-D MULTISAMPLING")
build_compute_shader(rt64 "src/shaders/LightingSkyAnalyzeCS.hlsl")
build_compute_shader(rt64 "src/shaders/LightingSkyAnalyzeCS.hlsl" "src/shaders/LightingSkyAnalyzeCSMS.hlsl" "-D MULTISAMPLING")

# They include the shared headers and helpers, so they're rebuilt whenever any of those changes.
file(GLOB RT64_LIGHTING_SKY_SHADER_DEPENDS "${PROJECT_SOURCE_DIR}/src/shaders/*.hlsli" "${PROJECT_SOURCE_DIR}/src/shared/*.h")
foreach(RT64_LIGHTING_SKY_SHADER LightingSkyLutPS LightingSkyPS LightingSkyPSMS LightingSkyAnalyzeCS LightingSkyAnalyzeCSMS)
    set(RT64_LIGHTING_SKY_SHADER_OUTPUT "${CMAKE_BINARY_DIR}/src/shaders/${RT64_LIGHTING_SKY_SHADER}.hlsl")
    add_custom_command(OUTPUT "${RT64_LIGHTING_SKY_SHADER_OUTPUT}.spv" APPEND DEPENDS ${RT64_LIGHTING_SKY_SHADER_DEPENDS})
    if (WIN32)
        add_custom_command(OUTPUT "${RT64_LIGHTING_SKY_SHADER_OUTPUT}.dxil" APPEND DEPENDS ${RT64_LIGHTING_SKY_SHADER_DEPENDS})
    endif()
endforeach()
