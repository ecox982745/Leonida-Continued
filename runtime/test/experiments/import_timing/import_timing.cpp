// Standalone Windows Vulkan compute probe. Run from this directory in an x64 MSVC developer shell.
//   glslangValidator -V --target-env vulkan1.2 stream.comp -o stream.spv
//   cl /nologo /std:c++20 /EHsc /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /I../.. /I"%VULKAN_SDK%/Include" import_timing.cpp
//   import_timing.exe stream.spv 24 128
// The last two arguments select measured rounds and buffer size in MiB. Each timed read immediately
// follows a write to that buffer. Copies and CPU checks run outside the timestamp interval.

#include "vk_test_device.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define VKD(name) vk.fn<PFN_##name>(#name)
#define VKI(name) vk.instance_fn<PFN_##name>(#name)
static void require(bool condition, const char *what) {
    if (!condition)
        throw std::runtime_error(what);
}
static void checked(VkResult result, const char *what) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(what) + " VkResult=" + std::to_string(result));
}
constexpr uint32_t lanes = 256, loads = 16;

constexpr auto handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
enum Placement { local, imported, ordinaryHost };
const char *names[] = {"device", "imported", "ordinary-host"};

struct Buffer {
    vktest::Device &vk;
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void *region{};
    explicit Buffer(vktest::Device &vk) : vk(vk) {}
    ~Buffer() {
        if (buffer)
            VKD(vkDestroyBuffer)(vk.device, buffer, nullptr);
        if (memory)
            VKD(vkFreeMemory)(vk.device, memory, nullptr);
        if (region)
            VirtualFree(region, 0, MEM_RELEASE);
    }
    void create(Placement placement, VkDeviceSize size, int sameType = -1) {
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ext.handleTypes = handleType;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.pNext = placement == imported ? &ext : nullptr;
        ci.size = size;
        ci.usage = usage;
        checked(VKD(vkCreateBuffer)(vk.device, &ci, nullptr, &buffer), "create buffer");
        VkMemoryRequirements req{};
        VKD(vkGetBufferMemoryRequirements)(vk.device, buffer, &req);
        VkPhysicalDeviceMemoryProperties props{};
        VKI(vkGetPhysicalDeviceMemoryProperties)(vk.physical, &props);
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.buffer = buffer;
        VkImportMemoryHostPointerInfoEXT importInfo{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        auto mask = req.memoryTypeBits;
        VkDeviceSize allocSize = req.size;
        if (placement == imported) {
            VkPhysicalDeviceExternalBufferInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
            query.handleType = handleType;
            query.usage = usage;
            VkExternalBufferProperties result{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
            VKI(vkGetPhysicalDeviceExternalBufferProperties)(vk.physical, &query, &result);
            require(result.externalMemoryProperties.externalMemoryFeatures &
                        VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
                    "unsupported import");
            VkDeviceSize align = vk.host_pointer_alignment;
            require(align && !(align & (align - 1)), "invalid import alignment");
            allocSize = (allocSize + align - 1) & ~(align - 1);
            region = VirtualAlloc(nullptr, size_t(allocSize), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            require(region && uintptr_t(region) % align == 0, "unaligned host allocation");
            std::memset(region, 0, size_t(allocSize));
            VkMemoryHostPointerPropertiesEXT host{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
            checked(VKD(vkGetMemoryHostPointerPropertiesEXT)(vk.device, handleType, region, &host),
                    "query host pointer");
            mask &= host.memoryTypeBits;
            importInfo.handleType = handleType;
            importInfo.pHostPointer = region;
            if (result.externalMemoryProperties.externalMemoryFeatures &
                VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT)
                importInfo.pNext = &dedicated;
        }
        const auto wanted = placement == local
                                ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            const auto mt = props.memoryTypes[i];
            bool onDevice = (props.memoryHeaps[mt.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            if ((mask & (1u << i)) && (mt.propertyFlags & wanted) == wanted && onDevice == (placement == local) &&
                (sameType < 0 || sameType == int(i))) {
                type = i;
                break;
            }
        }
        require(type >= 0, "matching memory type unavailable");
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        if (placement == imported)
            flags.pNext = &importInfo;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        ai.allocationSize = allocSize;
        ai.memoryTypeIndex = uint32_t(type);
        checked(VKD(vkAllocateMemory)(vk.device, &ai, nullptr, &memory), "allocate buffer memory");
        checked(VKD(vkBindBufferMemory)(vk.device, buffer, memory, 0), "bind buffer");
        const auto mt = props.memoryTypes[type];
        std::printf("%s: type=%d flags=0x%x heap=%u bytes=%llu\n", names[placement], type, mt.propertyFlags,
                    mt.heapIndex, static_cast<unsigned long long>(size));
    }
    int type = -1;
};
struct State {
    vktest::Device &vk;
    VkShaderModule shader{};
    VkPipelineLayout layout{};
    VkPipeline pipeline{};
    VkQueryPool queries{};
    ~State() {
        if (queries)
            VKD(vkDestroyQueryPool)(vk.device, queries, nullptr);
        if (pipeline)
            VKD(vkDestroyPipeline)(vk.device, pipeline, nullptr);
        if (layout)
            VKD(vkDestroyPipelineLayout)(vk.device, layout, nullptr);
        if (shader)
            VKD(vkDestroyShaderModule)(vk.device, shader, nullptr);
    }
};
struct Params {
    VkDeviceAddress data, checksums;
    uint32_t seed, writeMode;
};
static void barrier(vktest::Device &vk, VkCommandBuffer cmd, bool toHost = false) {
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    memory.dstAccessMask =
        toHost ? VK_ACCESS_HOST_READ_BIT : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VKD(vkCmdPipelineBarrier)(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              toHost ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                              &memory, 0, nullptr, 0, nullptr);
}
static uint32_t pattern(uint32_t i, uint32_t seed) { return i * 747796405u + seed; }
static void measure(const char *shaderPath, int rounds, uint32_t mib) {
    const uint32_t groups = mib * 64;
    const uint32_t sums = groups * lanes, words = sums * loads;
    const VkDeviceSize bytes = VkDeviceSize(words) * 4;
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    require(vk.create() && vk.buffer_device_address && vk.host_pointer_alignment, "Vulkan device/features");
    VkPhysicalDeviceProperties props{};
    VKI(vkGetPhysicalDeviceProperties)(vk.physical, &props);
    uint32_t count = 0;
    VKI(vkGetPhysicalDeviceQueueFamilyProperties)(vk.physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    VKI(vkGetPhysicalDeviceQueueFamilyProperties)(vk.physical, &count, families.data());
    auto validBits = families.at(vk.family).timestampValidBits;
    require(validBits && props.limits.timestampComputeAndGraphics && props.limits.timestampPeriod > 0,
            "GPU timestamps unsupported");
    std::printf("%s; timestamp %.3f ns, %u bits; validation %s\n", props.deviceName, props.limits.timestampPeriod,
                validBits, vk.validation ? "on" : "off");
    Buffer device(vk), host(vk), ordinary(vk);
    device.create(local, bytes);
    host.create(imported, bytes);
    ordinary.create(ordinaryHost, bytes, host.type);
    std::array<Buffer *, 3> buffers{&device, &host, &ordinary};
    VkBuffer output = vk.buffer(VkDeviceSize(sums) * 4,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, nullptr);
    void *mapped = nullptr;
    VkBuffer readback = vk.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    require(output && readback && mapped, "readback setup");
    std::ifstream file(shaderPath, std::ios::binary | std::ios::ate);
    auto length = std::streamoff(file.tellg());
    require(length > 0 && length % 4 == 0, "read SPIR-V");
    std::vector<uint32_t> code(size_t(length) / 4);
    file.seekg(0);
    require(bool(file.read(reinterpret_cast<char *>(code.data()), length)), "read SPIR-V bytes");
    State state{vk};
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = size_t(length);
    sm.pCode = code.data();
    checked(VKD(vkCreateShaderModule)(vk.device, &sm, nullptr, &state.shader), "shader module");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &range;
    checked(VKD(vkCreatePipelineLayout)(vk.device, &li, nullptr, &state.layout), "pipeline layout");
    VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pi.layout = state.layout;
    pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pi.stage.module = state.shader;
    pi.stage.pName = "main";
    checked(VKD(vkCreateComputePipelines)(vk.device, VK_NULL_HANDLE, 1, &pi, nullptr, &state.pipeline),
            "compute pipeline");
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 2;
    checked(VKD(vkCreateQueryPool)(vk.device, &qi, nullptr, &state.queries), "timestamp queries");
    auto sample = [&](int placement, unsigned write, uint32_t seed) {
        Params p{vk.device_address(buffers[placement]->buffer), vk.device_address(output), seed, write};
        require(p.data && p.checksums, "buffer device addresses");
        require(vk.run([&](VkCommandBuffer cmd) {
            barrier(vk, cmd);
            VKD(vkCmdResetQueryPool)(cmd, state.queries, 0, 2);
            VKD(vkCmdBindPipeline)(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, state.pipeline);
            VKD(vkCmdPushConstants)(cmd, state.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            VKD(vkCmdWriteTimestamp)(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, state.queries, 0);
            VKD(vkCmdDispatch)(cmd, groups, 1, 1);
            VKD(vkCmdWriteTimestamp)(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, state.queries, 1);
            return true;
        }),
                "submit dispatch");
        uint64_t t[2]{};
        checked(VKD(vkGetQueryPoolResults)(vk.device, state.queries, 0, 2, sizeof(t), t, sizeof(t[0]),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                "timestamp read");
        uint64_t mask = validBits == 64 ? UINT64_MAX : (uint64_t(1) << validBits) - 1;
        return double((t[1] - t[0]) & mask) * props.limits.timestampPeriod * 1e-6;
    };
    auto check = [&](VkBuffer source, uint32_t seed, bool checkSums) {
        VkBufferCopy cp{0, 0, VkDeviceSize(checkSums ? sums : words) * 4};
        require(vk.run([&](VkCommandBuffer cmd) {
            barrier(vk, cmd);
            VKD(vkCmdCopyBuffer)(cmd, source, readback, 1, &cp);
            barrier(vk, cmd, true);
            return true;
        }),
                "submit readback");
        auto data = static_cast<const uint32_t *>(mapped);
        for (uint32_t i = 0; i < (checkSums ? sums : words); ++i) {
            uint32_t expected = 0;
            if (checkSums) {
                uint32_t base = (i / lanes) * lanes * loads + i % lanes;
                for (uint32_t j = 0; j < loads; ++j)
                    expected = expected * 33u + pattern(base + j * lanes, seed);
            } else
                expected = pattern(i, seed);
            require(data[i] == expected, "readback mismatch");
        }
    };
    // Cycle through all six permutations so each placement occupies each timing slot equally often.
    const int order[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {0, 2, 1}, {1, 0, 2}};
    for (int round = 0; round < rounds + 6; ++round) {
        for (int k : order[round % 6]) {
            uint32_t seed = 0x12345678u + round * 103u + k;
            double store = sample(k, 1, seed);
            double load = sample(k, 0, seed);
            if (round >= 6)
                std::printf("sample,%d,%s,%.6f,%.6f\n", round - 6, names[k], store, load);
            if (round == 5 || round == rounds + 5) {
                check(buffers[k]->buffer, seed, false);
                check(output, seed, true);
            }
        }
    }
    require(vk.validation_errors() == 0, "validation errors");
}
int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: import_timing shader.spv measured_rounds working_set_MiB\n");
        return 2;
    }
    try {
        int rounds = std::stoi(argv[2]);
        require(rounds > 0 && rounds <= 120 && rounds % 6 == 0,
                "use 6 to 120 measured rounds in multiples of six");
        int mib = std::stoi(argv[3]);
        require(mib > 0 && mib <= 512, "working set must be 1 to 512 MiB");
        measure(argv[1], rounds, static_cast<uint32_t>(mib));
        require(vktest::validation_error_count() == 0, "validation errors at destruction");
        std::puts("PASS full buffer and checksum verification");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
