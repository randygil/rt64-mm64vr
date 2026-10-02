//
// RT64
//

#include "rt64_raytracing_shader_cache.h"

#include <unordered_set>

#include "rt64_descriptor_sets.h"

#include "shaders/RaytracingLibrary.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/RaytracingLibrary.hlsl.dxil.h"
#endif

namespace RT64 {
    // Must be big enough to hold the largest payload used by the library (HitInfo).
    static const uint32_t MaxPayloadSize = 64;

    // The DXC version used to build the shaders lists every global variable in the interface of every entry point of a
    // library. This is invalid for incoming ray payloads and hit attributes, as each hit or miss shader can only have one
    // of each. This removes the ones that aren't used by the call graph of each entry point.
    static std::vector<uint32_t> removeUnusedRayInterfaceVariables(const uint32_t *words, size_t wordCount) {
        const uint32_t OpEntryPoint = 15;
        const uint32_t OpFunction = 54;
        const uint32_t OpFunctionEnd = 56;
        const uint32_t OpFunctionCall = 57;
        const uint32_t OpVariable = 59;
        const uint32_t StorageClassHitAttributeKHR = 5339;
        const uint32_t StorageClassIncomingRayPayloadKHR = 5342;
        const size_t HeaderSize = 5;
        std::vector<uint32_t> result(words, words + wordCount);
        if (wordCount < HeaderSize) {
            return result;
        }

        // Gather the variables that must be filtered, the ids referenced by each function and the functions they call.
        std::unordered_set<uint32_t> filteredVariables;
        std::unordered_map<uint32_t, std::unordered_set<uint32_t>> functionUses;
        std::unordered_map<uint32_t, std::vector<uint32_t>> functionCalls;
        uint32_t currentFunction = 0;
        for (size_t i = HeaderSize; i < wordCount;) {
            const uint32_t instructionWords = words[i] >> 16;
            const uint32_t opcode = words[i] & 0xFFFF;
            if ((instructionWords == 0) || ((i + instructionWords) > wordCount)) {
                return result;
            }

            if ((opcode == OpVariable) && (instructionWords >= 4)) {
                const uint32_t storageClass = words[i + 3];
                if ((storageClass == StorageClassHitAttributeKHR) || (storageClass == StorageClassIncomingRayPayloadKHR)) {
                    filteredVariables.insert(words[i + 2]);
                }
            }
            else if ((opcode == OpFunction) && (instructionWords >= 3)) {
                currentFunction = words[i + 2];
            }
            else if (opcode == OpFunctionEnd) {
                currentFunction = 0;
            }

            if (currentFunction != 0) {
                std::unordered_set<uint32_t> &uses = functionUses[currentFunction];
                for (uint32_t w = 1; w < instructionWords; w++) {
                    uses.insert(words[i + w]);
                }

                if ((opcode == OpFunctionCall) && (instructionWords >= 4)) {
                    functionCalls[currentFunction].push_back(words[i + 3]);
                }
            }

            i += instructionWords;
        }

        if (filteredVariables.empty()) {
            return result;
        }

        auto gatherUses = [&](uint32_t entryFunction) {
            std::unordered_set<uint32_t> used;
            std::unordered_set<uint32_t> visited;
            std::vector<uint32_t> stack = { entryFunction };
            while (!stack.empty()) {
                const uint32_t function = stack.back();
                stack.pop_back();
                if (!visited.insert(function).second) {
                    continue;
                }

                const auto &uses = functionUses[function];
                used.insert(uses.begin(), uses.end());
                for (uint32_t callee : functionCalls[function]) {
                    stack.push_back(callee);
                }
            }

            return used;
        };

        // Rebuild the module with the filtered entry points.
        result.assign(words, words + HeaderSize);
        for (size_t i = HeaderSize; i < wordCount;) {
            const uint32_t instructionWords = words[i] >> 16;
            const uint32_t opcode = words[i] & 0xFFFF;
            if ((opcode == OpEntryPoint) && (instructionWords >= 4)) {
                // Skip the execution model, the function and the null terminated name to find the interface.
                const uint32_t entryFunction = words[i + 2];
                uint32_t interfaceStart = 3;
                while (interfaceStart < instructionWords) {
                    const uint32_t nameWord = words[i + interfaceStart];
                    interfaceStart++;
                    if ((nameWord & 0xFF000000U) == 0) {
                        break;
                    }
                }

                const std::unordered_set<uint32_t> used = gatherUses(entryFunction);
                std::vector<uint32_t> instruction(words + i, words + i + interfaceStart);
                for (uint32_t w = interfaceStart; w < instructionWords; w++) {
                    const uint32_t id = words[i + w];
                    if ((filteredVariables.find(id) == filteredVariables.end()) || (used.find(id) != used.end())) {
                        instruction.push_back(id);
                    }
                }

                instruction[0] = (uint32_t(instruction.size()) << 16) | OpEntryPoint;
                result.insert(result.end(), instruction.begin(), instruction.end());
            }
            else {
                result.insert(result.end(), words + i, words + i + instructionWords);
            }

            i += instructionWords;
        }

        return result;
    }

    // RaytracingShaderCache

    RaytracingShaderCache::RaytracingShaderCache(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary) {
        assert(device != nullptr);
        assert(shaderLibrary != nullptr);

        this->device = device;
        this->shaderFormat = shaderFormat;
        this->shaderLibrary = shaderLibrary;
    }

    RaytracingShaderCache::~RaytracingShaderCache() {
        for (RaytracingState &state : states) {
            state.pipeline.reset();
        }

        libraryShader.reset();
        pipelineLayout.reset();
    }

    bool RaytracingShaderCache::isSetup() const {
        return setupDone;
    }

    void RaytracingShaderCache::setup() {
        assert(!setupDone);

        // The layout must match the one used by the framebuffer renderer for the raster ubershader so the same
        // descriptor sets can be used. It's a global layout bound by the command list before tracing rays.
        FramebufferRendererDescriptorCommonSet descriptorCommonSet(shaderLibrary->samplerLibrary, true);
        FramebufferRendererDescriptorTextureSet descriptorTextureSet;
        FramebufferRendererDescriptorFramebufferSet descriptorFramebufferSet;
        RenderPipelineLayoutBuilder layoutBuilder;
        layoutBuilder.begin(false);
        layoutBuilder.addDescriptorSet(descriptorCommonSet);
        layoutBuilder.addDescriptorSet(descriptorTextureSet);
        layoutBuilder.addDescriptorSet(descriptorTextureSet);
        layoutBuilder.addDescriptorSet(descriptorFramebufferSet);
        layoutBuilder.end();
        pipelineLayout = layoutBuilder.create(device);

        const void *libraryBlob = nullptr;
        uint64_t libraryBlobSize = 0;
        switch (shaderFormat) {
#   ifdef _WIN32
        case RenderShaderFormat::DXIL:
            libraryBlob = RaytracingLibraryBlobDXIL;
            libraryBlobSize = sizeof(RaytracingLibraryBlobDXIL);
            break;
#   endif
        case RenderShaderFormat::SPIRV:
            libraryBlob = RaytracingLibraryBlobSPIRV;
            libraryBlobSize = sizeof(RaytracingLibraryBlobSPIRV);
            break;
        default:
            assert(false && "Unsupported shader format for raytracing.");
            return;
        }

        if (shaderFormat == RenderShaderFormat::SPIRV) {
            libraryWords = removeUnusedRayInterfaceVariables(reinterpret_cast<const uint32_t *>(libraryBlob), size_t(libraryBlobSize / sizeof(uint32_t)));
            libraryBlob = libraryWords.data();
            libraryBlobSize = libraryWords.size() * sizeof(uint32_t);
        }

        libraryShader = device->createShader(libraryBlob, libraryBlobSize, nullptr, shaderFormat);

        typedef RenderRaytracingPipelineLibrarySymbolType SymbolType;
        const RenderRaytracingPipelineLibrarySymbol librarySymbols[] = {
            { "PrimaryRayGen", SymbolType::RAYGEN },
            { "DirectRayGen", SymbolType::RAYGEN },
            { "IndirectRayGen", SymbolType::RAYGEN },
            { "ReflectionRayGen", SymbolType::RAYGEN },
            { "RefractionRayGen", SymbolType::RAYGEN },
            { "SurfaceMiss", SymbolType::MISS },
            { "ShadowMiss", SymbolType::MISS },
            { "SurfaceAnyHit", SymbolType::ANY_HIT },
            { "SurfaceClosestHit", SymbolType::CLOSEST_HIT },
            { "ShadowAnyHit", SymbolType::ANY_HIT },
            { "ShadowClosestHit", SymbolType::CLOSEST_HIT }
        };

        const RenderRaytracingPipelineLibrary library(libraryShader.get(), librarySymbols, uint32_t(std::size(librarySymbols)));
        const RenderRaytracingPipelineHitGroup hitGroups[] = {
            { "SurfaceHitGroup", "SurfaceClosestHit", "SurfaceAnyHit" },
            { "ShadowHitGroup", "ShadowClosestHit", "ShadowAnyHit" }
        };

        RenderRaytracingPipelineDesc pipelineDesc;
        pipelineDesc.libraries = &library;
        pipelineDesc.librariesCount = 1;
        pipelineDesc.hitGroups = hitGroups;
        pipelineDesc.hitGroupsCount = uint32_t(std::size(hitGroups));
        pipelineDesc.pipelineLayout = pipelineLayout.get();
        pipelineDesc.maxPayloadSize = MaxPayloadSize;
        pipelineDesc.maxAttributeSize = 2 * sizeof(float);
        pipelineDesc.maxRecursionDepth = 1;

        for (RaytracingState &state : states) {
            state.pipeline = device->createRaytracingPipeline(pipelineDesc);
            state.rayGenPrograms = {
                state.pipeline->getProgram("PrimaryRayGen"),
                state.pipeline->getProgram("DirectRayGen"),
                state.pipeline->getProgram("IndirectRayGen"),
                state.pipeline->getProgram("ReflectionRayGen"),
                state.pipeline->getProgram("RefractionRayGen")
            };

            state.missPrograms = {
                state.pipeline->getProgram("SurfaceMiss"),
                state.pipeline->getProgram("ShadowMiss")
            };

            RaytracingShaderPrograms uberPrograms;
            uberPrograms.surface = state.pipeline->getProgram("SurfaceHitGroup");
            uberPrograms.shadow = state.pipeline->getProgram("ShadowHitGroup");
            state.shaderProgramsMap[UberShaderHash] = uberPrograms;
        }

        setupDone = true;
    }

    void RaytracingShaderCache::submit(const ShaderDescription &desc) {
        // Specialized hit groups are not generated yet. Every draw call uses the ubershader hit groups.
    }

    void RaytracingShaderCache::setNextState() {
        activeState = (activeState + 1) % StateCount;
    }

    int32_t RaytracingShaderCache::getActiveState() const {
        return activeState;
    }
};
