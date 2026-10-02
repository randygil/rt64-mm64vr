# Shaders of the post effects (render/rt64_post_effects.cpp). Included from CMakeLists.txt after the shader build functions.
build_pixel_shader(rt64 "src/shaders/PostEffectsPrefilterPS.hlsl")
build_pixel_shader(rt64 "src/shaders/PostEffectsPrefilterPS.hlsl" "src/shaders/PostEffectsPrefilterPSMS.hlsl" "-D MULTISAMPLING")
build_pixel_shader(rt64 "src/shaders/PostEffectsDownsamplePS.hlsl")
build_pixel_shader(rt64 "src/shaders/PostEffectsUpsamplePS.hlsl")
build_pixel_shader(rt64 "src/shaders/PostEffectsShaftsPS.hlsl")
build_pixel_shader(rt64 "src/shaders/PostEffectsComposePS.hlsl")

# They include the parameters shared with the C++ side (PostEffectsParams.hlsli) and other helpers, so they're rebuilt
# whenever any of those change.
file(GLOB RT64_POST_EFFECTS_SHADER_DEPENDS "${PROJECT_SOURCE_DIR}/src/shaders/*.hlsli" "${PROJECT_SOURCE_DIR}/src/shared/*.h")
foreach(RT64_POST_EFFECTS_SHADER PostEffectsPrefilterPS PostEffectsPrefilterPSMS PostEffectsDownsamplePS PostEffectsUpsamplePS PostEffectsShaftsPS PostEffectsComposePS)
    set(RT64_POST_EFFECTS_SHADER_OUTPUT "${CMAKE_BINARY_DIR}/src/shaders/${RT64_POST_EFFECTS_SHADER}.hlsl")
    add_custom_command(OUTPUT "${RT64_POST_EFFECTS_SHADER_OUTPUT}.spv" APPEND DEPENDS ${RT64_POST_EFFECTS_SHADER_DEPENDS})
    if (WIN32)
        add_custom_command(OUTPUT "${RT64_POST_EFFECTS_SHADER_OUTPUT}.dxil" APPEND DEPENDS ${RT64_POST_EFFECTS_SHADER_DEPENDS})
    endif()
endforeach()
