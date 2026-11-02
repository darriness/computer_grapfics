#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "application.hpp"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>     
#include <algorithm>   

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

namespace application {

namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

struct Material {
    Texture albedo;
    Texture specular;
    Texture emissive;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct ModelInstance {
    glm::vec3 position = {0,0,0};
    glm::vec3 rotation = {0,0,0};
    glm::vec3 scale = {1,1,1};
    int material_index = 0;
    bool enable_nontrivial = false;
    bool is_static = false;
};

struct PipelineConsts {
    int enable_nontrivial;
    float time;
};

struct SceneUniforms {
    glm::mat4 view_projection;
    glm::vec4 camera_position;
    glm::vec4 dir_light_direction;
    glm::vec4 dir_light_color;
    glm::vec4 ambient_light;
    int point_light_count;
    int spot_light_count;
    float padding[2];
};

struct ModelUniforms {
    glm::mat4 model;
    glm::vec4 albedo;
    glm::vec4 specular;
    float shininess;
    float padding[3];
};

struct PointLight {
    glm::vec4 position;
    glm::vec4 color;
};

struct SpotLight {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 color;
    float inner_angle;
    float outer_angle;
    float padding[2];
};

struct LightPushConstants {
    glm::mat4 mvp;
    glm::vec4 color;
};

// --- Глобальные переменные ---
VkCommandPool g_command_pool = VK_NULL_HANDLE;

VkBuffer vk_cube_vb, vk_cube_ib;
VmaAllocation vk_cube_vb_alloc, vk_cube_ib_alloc;
uint32_t cube_index_count = 0;

VkBuffer vk_light_cube_vb, vk_light_cube_ib;
VmaAllocation vk_light_cube_vb_alloc, vk_light_cube_ib_alloc;

VkBuffer vk_cone_vb;
VmaAllocation vk_cone_vb_alloc;

VkBuffer vk_scene_ubo;
VmaAllocation vk_scene_ubo_alloc;
SceneUniforms* scene_ubo_mem = nullptr;

VkBuffer vk_model_ubo;
VmaAllocation vk_model_ubo_alloc;
uint8_t* vk_model_ubo_mem = nullptr;
VkDeviceSize model_ubo_aligned_size = 0;
const uint32_t max_models = 1024;

VkBuffer vk_point_ssbo, vk_spot_ssbo;
VmaAllocation vk_point_ssbo_alloc, vk_spot_ssbo_alloc;
void* point_ssbo_mem = nullptr;
void* spot_ssbo_mem = nullptr;

VkDescriptorSetLayout vk_scene_dsl, vk_model_dsl, vk_material_dsl;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_ds, vk_model_ds;
VkPipelineLayout vk_pipeline_layout, vk_light_pipeline_layout;
VkPipeline vk_main_pipeline, vk_light_pipeline, vk_line_pipeline;

std::vector<Material> g_materials;
std::vector<ModelInstance> g_models;
float g_time = 0.0f;

struct AppState {
    glm::vec3 camera_pos = {3.0f, 3.0f, 5.0f};
    float camera_yaw = -135.0f;
    float camera_pitch = -30.0f;
    float fov = 60.0f;

    glm::vec3 albedo = {1.0f, 1.0f, 1.0f};
    glm::vec3 specular = {1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;

    glm::vec3 ambient = {0.2f, 0.2f, 0.2f};
    glm::vec3 dir_light_dir = {-0.5f, -1.0f, -0.3f};
    glm::vec3 dir_light_color = {1.0f, 1.0f, 1.0f};
    float dir_light_intensity = 1.0f;

    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;

    bool show_light_visuals = true;
    bool animate_scene = true;
    float anim_speed = 1.0f;
} state;

// --- Вспомогательные функции ---
VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = g_command_pool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(graphics::internal::context.device, &allocInfo, &commandBuffer);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(graphics::internal::context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics::internal::context.graphics_queue);
    vkFreeCommandBuffers(graphics::internal::context.device, g_command_pool, 1, &commandBuffer);
}

void createDeviceLocalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation, const void* data) {
    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    VmaAllocationInfo stagingInfo;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, &stagingInfo);
    memcpy(stagingInfo.pMappedData, data, size);

    bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    allocInfo.flags = 0;
    vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
    endSingleTimeCommands(cmd);

    vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
}

void createTextureFromData(const void* pixels, int width, int height, VkFormat format, Texture& tex) {
    VkDeviceSize imageSize = width * height * 4;
    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    VmaAllocationInfo stagingInfo;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, &stagingInfo);
    memcpy(stagingInfo.pMappedData, pixels, imageSize);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo imageAllocInfo{};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vmaCreateImage(graphics::internal::context.allocator, &imageInfo, &imageAllocInfo, &tex.image, &tex.allocation, nullptr);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(cmd);
    vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    vkCreateImageView(graphics::internal::context.device, &viewInfo, nullptr, &tex.view);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkCreateSampler(graphics::internal::context.device, &samplerInfo, nullptr, &tex.sampler);
}

bool loadTextureFromFile(const char* path, Texture& tex, VkFormat format = VK_FORMAT_R8G8B8A8_SRGB) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "Failed to load texture: " << path << std::endl;
        return false;
    }
    createTextureFromData(pixels, width, height, format, tex);
    stbi_image_free(pixels);
    return true;
}

void createBlackTexture(Texture& tex) {
    uint8_t pixel[4] = {0, 0, 0, 255};
    createTextureFromData(pixel, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, tex);
}

void generateProceduralSpecular(int width, int height, Texture& tex) {
    std::vector<uint8_t> pixels(width * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx = (float)x / width - 0.5f;
            float fy = (float)y / height - 0.5f;
            float dist = sqrt(fx*fx + fy*fy);
            float value = glm::clamp(1.0f - dist * 2.0f, 0.0f, 1.0f);
            value = pow(value, 4.0f);
            uint8_t v = (uint8_t)(value * 255);
            int idx = (y * width + x) * 4;
            pixels[idx + 0] = v;
            pixels[idx + 1] = v;
            pixels[idx + 2] = v;
            pixels[idx + 3] = 255;
        }
    }
    createTextureFromData(pixels.data(), width, height, VK_FORMAT_R8G8B8A8_UNORM, tex);
}

void generateProceduralEmissive(int width, int height, Texture& tex) {
    std::vector<uint8_t> pixels(width * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float fx = (float)x / width * 10.0f;
            float fy = (float)y / height * 10.0f;
            float noise = sin(fx) * cos(fy) + sin(fx*2.3f) * cos(fy*1.7f);
            noise = (noise + 2.0f) / 4.0f;
            uint8_t r = (uint8_t)(noise * 255);
            uint8_t g = (uint8_t)(noise * 200);
            uint8_t b = (uint8_t)(noise * 155);
            int idx = (y * width + x) * 4;
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255;
        }
    }
    createTextureFromData(pixels.data(), width, height, VK_FORMAT_R8G8B8A8_UNORM, tex);
}

void createSolidColorMaterial(glm::vec4 color, Material& mat) {
    uint8_t pixels[4] = {
        static_cast<uint8_t>(color.r * 255.0f),
        static_cast<uint8_t>(color.g * 255.0f),
        static_cast<uint8_t>(color.b * 255.0f),
        255
    };
    // Создаем текстуру 1x1
    createTextureFromData(pixels, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, mat.albedo);
    
    // Для specular и emissive используем черный цвет (чтобы не было бликов и свечения)
    createBlackTexture(mat.specular);
    createBlackTexture(mat.emissive);
}

void createMaterial(const char* albedoPath, Material& mat, bool useSpecular = false, bool useEmissive = true) {
    if (!loadTextureFromFile(albedoPath, mat.albedo)) {
        int w = 256, h = 256;
        std::vector<uint8_t> pixels(w * h * 4);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = (y * w + x) * 4;
                bool checker = ((x / 32) + (y / 32)) % 2 == 0;
                uint8_t c = checker ? 200 : 100;
                pixels[idx + 0] = c;
                pixels[idx + 1] = c;
                pixels[idx + 2] = c;
                pixels[idx + 3] = 255;
            }
        }
        createTextureFromData(pixels.data(), w, h, VK_FORMAT_R8G8B8A8_SRGB, mat.albedo);
    }
    if (useSpecular) {
        generateProceduralSpecular(256, 256, mat.specular);
    } else {
        createBlackTexture(mat.specular);
    }
    if (useEmissive) {
        generateProceduralEmissive(256, 256, mat.emissive);
    } else {
        createBlackTexture(mat.emissive);
    }
}

VkDescriptorSet createMaterialDescriptorSet(const Material& mat) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = vk_descriptor_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &vk_material_dsl;
    VkDescriptorSet ds;
    vkAllocateDescriptorSets(graphics::internal::context.device, &allocInfo, &ds);

    VkDescriptorImageInfo albedoInfo{ mat.albedo.sampler, mat.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo specularInfo{ mat.specular.sampler, mat.specular.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo emissiveInfo{ mat.emissive.sampler, mat.emissive.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ds;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &albedoInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ds;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &specularInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = ds;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &emissiveInfo;

    vkUpdateDescriptorSets(graphics::internal::context.device, 3, writes, 0, nullptr);
    return ds;
}

void generateCube(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    vertices.push_back({{-1, -1,  1}, { 0,  0,  1}, {0, 1}});
    vertices.push_back({{ 1, -1,  1}, { 0,  0,  1}, {1, 1}});
    vertices.push_back({{ 1,  1,  1}, { 0,  0,  1}, {1, 0}});
    vertices.push_back({{-1,  1,  1}, { 0,  0,  1}, {0, 0}});

    vertices.push_back({{ 1, -1, -1}, { 0,  0, -1}, {0, 1}});
    vertices.push_back({{-1, -1, -1}, { 0,  0, -1}, {1, 1}});
    vertices.push_back({{-1,  1, -1}, { 0,  0, -1}, {1, 0}});
    vertices.push_back({{ 1,  1, -1}, { 0,  0, -1}, {0, 0}});

    vertices.push_back({{-1,  1,  1}, { 0,  1,  0}, {0, 1}});
    vertices.push_back({{ 1,  1,  1}, { 0,  1,  0}, {1, 1}});
    vertices.push_back({{ 1,  1, -1}, { 0,  1,  0}, {1, 0}});
    vertices.push_back({{-1,  1, -1}, { 0,  1,  0}, {0, 0}});

    vertices.push_back({{-1, -1, -1}, { 0, -1,  0}, {0, 1}});
    vertices.push_back({{ 1, -1, -1}, { 0, -1,  0}, {1, 1}});
    vertices.push_back({{ 1, -1,  1}, { 0, -1,  0}, {1, 0}});
    vertices.push_back({{-1, -1,  1}, { 0, -1,  0}, {0, 0}});

    vertices.push_back({{ 1, -1,  1}, { 1,  0,  0}, {0, 1}});
    vertices.push_back({{ 1, -1, -1}, { 1,  0,  0}, {1, 1}});
    vertices.push_back({{ 1,  1, -1}, { 1,  0,  0}, {1, 0}});
    vertices.push_back({{ 1,  1,  1}, { 1,  0,  0}, {0, 0}});

    vertices.push_back({{-1, -1, -1}, {-1,  0,  0}, {0, 1}});
    vertices.push_back({{-1, -1,  1}, {-1,  0,  0}, {1, 1}});
    vertices.push_back({{-1,  1,  1}, {-1,  0,  0}, {1, 0}});
    vertices.push_back({{-1,  1, -1}, {-1,  0,  0}, {0, 0}});

    for (int i = 0; i < 6; ++i) {
        uint32_t offset = i * 4;
        indices.push_back(offset + 0);
        indices.push_back(offset + 1);
        indices.push_back(offset + 2);
        indices.push_back(offset + 2);
        indices.push_back(offset + 3);
        indices.push_back(offset + 0);
    }
}

void generateSpotCone(std::vector<Vertex>& vertices) {
    float radius = 0.5f;
    float height = 2.0f;
    int segments = 16;
    glm::vec3 tip(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < segments; ++i) {
        float angle = (float)i / segments * 2.0f * 3.14159265f;
        float x = radius * cos(angle);
        float y = radius * sin(angle);
        glm::vec3 base(x, y, -height);
        vertices.push_back({tip, {0,0,0}, {0,0}});
        vertices.push_back({base, {0,0,0}, {0,0}});
    }
    for (int i = 0; i < segments; ++i) {
        float a1 = (float)i / segments * 2.0f * 3.14159265f;
        float a2 = (float)(i + 1) / segments * 2.0f * 3.14159265f;
        glm::vec3 p1(radius * cos(a1), radius * sin(a1), -height);
        glm::vec3 p2(radius * cos(a2), radius * sin(a2), -height);
        vertices.push_back({p1, {0,0,0}, {0,0}});
        vertices.push_back({p2, {0,0,0}, {0,0}});
    }
}

VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0); file.read(buffer.data(), size);
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = reinterpret_cast<const uint32_t*>(buffer.data());
    VkShaderModule module = VK_NULL_HANDLE;
    vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module);
    return module;
}

VkPipeline createPipeline(const char* vertPath, const char* fragPath, VkPipelineLayout layout, bool isMain, VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
    auto& ctx = graphics::internal::context;
    VkShaderModule vert = loadShaderModule(vertPath);
    VkShaderModule frag = loadShaderModule(fragPath);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[3] = {};
    attributes[0].location = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = offsetof(Vertex, position);
    attributes[1].location = 1;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = offsetof(Vertex, normal);
    attributes[2].location = 2;
    attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[2].offset = offsetof(Vertex, uv);

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = isMain ? 3 : 1;
    vertex_input.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = topology;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = (topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST) ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend_attachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = layout;
    pipeline_info.renderPass = ctx.render_pass;
    pipeline_info.subpass = 0;

    VkPipeline pipeline;
    vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);

    vkDestroyShaderModule(ctx.device, vert, nullptr);
    vkDestroyShaderModule(ctx.device, frag, nullptr);
    return pipeline;
}

bool createBuffers() {
    auto& ctx = graphics::internal::context;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = ctx.graphics_queue_index;
    if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &g_command_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool\n";
        return false;
    }

    std::vector<Vertex> cubeVertices;
    std::vector<uint32_t> cubeIndices;
    generateCube(cubeVertices, cubeIndices);
    cube_index_count = static_cast<uint32_t>(cubeIndices.size());
    createDeviceLocalBuffer(cubeVertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk_cube_vb, vk_cube_vb_alloc, cubeVertices.data());
    createDeviceLocalBuffer(cubeIndices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, vk_cube_ib, vk_cube_ib_alloc, cubeIndices.data());

    std::vector<Vertex> lightCubeVerts = {
        {{-1, -1, -1}, {0,0,0}, {0,0}}, {{ 1, -1, -1}, {0,0,0}, {0,0}}, {{ 1,  1, -1}, {0,0,0}, {0,0}}, {{-1,  1, -1}, {0,0,0}, {0,0}},
        {{-1, -1,  1}, {0,0,0}, {0,0}}, {{ 1, -1,  1}, {0,0,0}, {0,0}}, {{ 1,  1,  1}, {0,0,0}, {0,0}}, {{-1,  1,  1}, {0,0,0}, {0,0}}
    };
    std::vector<uint32_t> lightCubeIdx = {
        4, 5, 6,  4, 6, 7,  1, 0, 3,  1, 3, 2,  5, 1, 2,  5, 2, 6,
        0, 4, 7,  0, 7, 3,  3, 7, 6,  3, 6, 2,  1, 5, 4,  1, 4, 0
    };
    createDeviceLocalBuffer(lightCubeVerts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk_light_cube_vb, vk_light_cube_vb_alloc, lightCubeVerts.data());
    createDeviceLocalBuffer(lightCubeIdx.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, vk_light_cube_ib, vk_light_cube_ib_alloc, lightCubeIdx.data());

    std::vector<Vertex> coneVerts;
    generateSpotCone(coneVerts);
    createDeviceLocalBuffer(coneVerts.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk_cone_vb, vk_cone_vb_alloc, coneVerts.data());

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(SceneUniforms);
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VmaAllocationInfo ai;
    vmaCreateBuffer(ctx.allocator, &info, &allocInfo, &vk_scene_ubo, &vk_scene_ubo_alloc, &ai);
    scene_ubo_mem = static_cast<SceneUniforms*>(ai.pMappedData);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physical_device, &props);
    VkDeviceSize minAlign = props.limits.minUniformBufferOffsetAlignment;
    model_ubo_aligned_size = (sizeof(ModelUniforms) + minAlign - 1) & ~(minAlign - 1);

    info.size = model_ubo_aligned_size * max_models;
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &allocInfo, &vk_model_ubo, &vk_model_ubo_alloc, &ai);
    vk_model_ubo_mem = static_cast<uint8_t*>(ai.pMappedData);

    info.size = 100 * sizeof(PointLight);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &allocInfo, &vk_point_ssbo, &vk_point_ssbo_alloc, &ai);
    point_ssbo_mem = ai.pMappedData;

    info.size = 100 * sizeof(SpotLight);
    vmaCreateBuffer(ctx.allocator, &info, &allocInfo, &vk_spot_ssbo, &vk_spot_ssbo_alloc, &ai);
    spot_ssbo_mem = ai.pMappedData;

    return true;
}

bool createDescriptorsAndPipelines() {
    auto& ctx = graphics::internal::context;

    // Set 0: Scene
    VkDescriptorSetLayoutBinding scene_bindings[3] = {};
    scene_bindings[0].binding = 0;
    scene_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    scene_bindings[0].descriptorCount = 1;
    scene_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    scene_bindings[1].binding = 1;
    scene_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    scene_bindings[1].descriptorCount = 1;
    scene_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    scene_bindings[2].binding = 2;
    scene_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    scene_bindings[2].descriptorCount = 1;
    scene_bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_scene{};
    dsl_scene.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_scene.bindingCount = 3;
    dsl_scene.pBindings = scene_bindings;
    vkCreateDescriptorSetLayout(ctx.device, &dsl_scene, nullptr, &vk_scene_dsl);

    // Set 1: Model (Dynamic UBO)
    VkDescriptorSetLayoutBinding model_binding{};
    model_binding.binding = 0;
    model_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    model_binding.descriptorCount = 1;
    model_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_model{};
    dsl_model.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_model.bindingCount = 1;
    dsl_model.pBindings = &model_binding;
    vkCreateDescriptorSetLayout(ctx.device, &dsl_model, nullptr, &vk_model_dsl);

    // Set 2: Material
    VkDescriptorSetLayoutBinding material_bindings[3] = {};
    material_bindings[0].binding = 0;
    material_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    material_bindings[0].descriptorCount = 1;
    material_bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    material_bindings[1].binding = 1;
    material_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    material_bindings[1].descriptorCount = 1;
    material_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    material_bindings[2].binding = 2;
    material_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    material_bindings[2].descriptorCount = 1;
    material_bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_material{};
    dsl_material.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_material.bindingCount = 3;
    dsl_material.pBindings = material_bindings;
    vkCreateDescriptorSetLayout(ctx.device, &dsl_material, nullptr, &vk_material_dsl);

    // Пул дескрипторов
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32 }
    };
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 16;
    pool_info.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]);
    pool_info.pPoolSizes = pool_sizes;
    vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &vk_descriptor_pool);

    // Аллокация сетов
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = vk_descriptor_pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &vk_scene_dsl;
    vkAllocateDescriptorSets(ctx.device, &alloc, &vk_scene_ds);

    alloc.pSetLayouts = &vk_model_dsl;
    vkAllocateDescriptorSets(ctx.device, &alloc, &vk_model_ds);

    // Запись дескрипторов
    VkDescriptorBufferInfo scene_ubo_info{vk_scene_ubo, 0, sizeof(SceneUniforms)};
    VkDescriptorBufferInfo point_ssbo_info{vk_point_ssbo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo spot_ssbo_info{vk_spot_ssbo, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo model_ubo_info{vk_model_ubo, 0, sizeof(ModelUniforms)};

    VkWriteDescriptorSet writes[4] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = vk_scene_ds; writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1; writes[0].pBufferInfo = &scene_ubo_info;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = vk_scene_ds; writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1; writes[1].pBufferInfo = &point_ssbo_info;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = vk_scene_ds; writes[2].dstBinding = 2;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1; writes[2].pBufferInfo = &spot_ssbo_info;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = vk_model_ds; writes[3].dstBinding = 0;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[3].descriptorCount = 1; writes[3].pBufferInfo = &model_ubo_info;

    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);

    // Pipeline layout
    VkDescriptorSetLayout layouts[] = { vk_scene_dsl, vk_model_dsl, vk_material_dsl };
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PipelineConsts);

    VkPipelineLayoutCreateInfo pl_info{};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 3;
    pl_info.pSetLayouts = layouts;
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &push_range;
    vkCreatePipelineLayout(ctx.device, &pl_info, nullptr, &vk_pipeline_layout);

    vk_main_pipeline = createPipeline("shaders/main.vert.spv", "shaders/main.frag.spv", vk_pipeline_layout, true);

    VkPushConstantRange light_push_range{};
    light_push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    light_push_range.offset = 0;
    light_push_range.size = sizeof(LightPushConstants);

    VkPipelineLayoutCreateInfo light_pl_info{};
    light_pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    light_pl_info.pushConstantRangeCount = 1;
    light_pl_info.pPushConstantRanges = &light_push_range;
    vkCreatePipelineLayout(ctx.device, &light_pl_info, nullptr, &vk_light_pipeline_layout);

    vk_light_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", vk_light_pipeline_layout, false);
    vk_line_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", vk_light_pipeline_layout, false, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

    Material mat1;
    createMaterial("textures/cat1.png", mat1, true, true);
    mat1.descriptor_set = createMaterialDescriptorSet(mat1);
    g_materials.push_back(mat1);

    Material mat2;
    createMaterial("textures/cat2.png", mat2, false, false);
    mat2.descriptor_set = createMaterialDescriptorSet(mat2);
    g_materials.push_back(mat2);

    Material mat3;
    createMaterial("textures/cat3.png", mat3, true, true);
    mat3.descriptor_set = createMaterialDescriptorSet(mat3);
    g_materials.push_back(mat3);

    // 4. Красный пол
    Material matFloor;
    createSolidColorMaterial({1.0f, 0.0f, 0.0f, 1.0f}, matFloor); // Ярко-красный
    matFloor.descriptor_set = createMaterialDescriptorSet(matFloor);
    g_materials.push_back(matFloor);

    // 5. Белые стены
    Material matWall;
    createSolidColorMaterial({0.9f, 0.9f, 0.9f, 1.0f}, matWall); // Слегка серовато-белый (чтобы не слепил глаза)
    matWall.descriptor_set = createMaterialDescriptorSet(matWall);
    g_materials.push_back(matWall);

    return true;
}

} // namespace

bool initialize() {
    if (!createBuffers()) return false;
    if (!createDescriptorsAndPipelines()) return false;

    //state.point_lights.push_back({{2.0f, 1.0f, 2.0f, 15.0f}, {1.0f, 0.2f, 0.2f, 1.0f}});
    //state.point_lights.push_back({{-2.0f, 1.0f, 2.0f, 15.0f}, {0.2f, 0.2f, 1.0f, 1.0f}});
    //state.spot_lights.push_back({{0.0f, 3.0f, 0.0f, 30.0f}, {0.0f, -1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, 15.0f, 30.0f});

    ModelInstance m1; m1.position = {0, 0, 0}; m1.material_index = 0; m1.enable_nontrivial = true;
    g_models.push_back(m1);
    ModelInstance m2; m2.position = {3, 0, 0}; m2.material_index = 1; m2.enable_nontrivial = true;
    g_models.push_back(m2);
    ModelInstance m3; m3.position = {-3, 0, 0}; m3.material_index = 2; m3.enable_nontrivial = true;
    g_models.push_back(m3);

        // --- Создаем комнату ---
    
    // Пол (сплющенный куб)
    ModelInstance floor;
    floor.position = {0.0f, -1.1f, 0.0f}; // Чуть ниже котов
    floor.scale = {15.0f, 0.1f, 15.0f};   // Широкий и плоский
    floor.material_index = 3;             // Индекс красного материала
    floor.is_static = true;
    g_models.push_back(floor);

    // Задняя стена
    ModelInstance wallBack;
    wallBack.position = {0.0f, 5.0f, -8.0f};
    wallBack.scale = {15.0f, 10.0f, 0.1f};
    wallBack.material_index = 4;          // Индекс белого материала
    wallBack.is_static = true;
    g_models.push_back(wallBack);

    // Передняя стена
    ModelInstance wallFront;
    wallFront.position = {0.0f, 5.0f, 8.0f};
    wallFront.scale = {15.0f, 10.0f, 0.1f};
    wallFront.material_index = 4;
    wallFront.is_static = true;
    g_models.push_back(wallFront);

    // Левая стена
    ModelInstance wallLeft;
    wallLeft.position = {-8.0f, 5.0f, 0.0f};
    wallLeft.scale = {0.1f, 10.0f, 15.0f};
    wallLeft.material_index = 4;
    wallLeft.is_static = true;
    g_models.push_back(wallLeft);

    // Правая стена
    ModelInstance wallRight;
    wallRight.position = {8.0f, 5.0f, 0.0f};
    wallRight.scale = {0.1f, 10.0f, 15.0f};
    wallRight.material_index = 4;
    wallRight.is_static = true;
    g_models.push_back(wallRight);


    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);

    for (auto& mat : g_materials) {
        vkDestroySampler(ctx.device, mat.albedo.sampler, nullptr);
        vkDestroyImageView(ctx.device, mat.albedo.view, nullptr);
        vmaDestroyImage(ctx.allocator, mat.albedo.image, mat.albedo.allocation);

        vkDestroySampler(ctx.device, mat.specular.sampler, nullptr);
        vkDestroyImageView(ctx.device, mat.specular.view, nullptr);
        vmaDestroyImage(ctx.allocator, mat.specular.image, mat.specular.allocation);

        vkDestroySampler(ctx.device, mat.emissive.sampler, nullptr);
        vkDestroyImageView(ctx.device, mat.emissive.view, nullptr);
        vmaDestroyImage(ctx.allocator, mat.emissive.image, mat.emissive.allocation);
    }

    vkDestroyPipeline(ctx.device, vk_main_pipeline, nullptr);
    vkDestroyPipeline(ctx.device, vk_light_pipeline, nullptr);
    vkDestroyPipeline(ctx.device, vk_line_pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(ctx.device, vk_light_pipeline_layout, nullptr);

    vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_scene_dsl, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_model_dsl, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_material_dsl, nullptr);

    vmaDestroyBuffer(ctx.allocator, vk_scene_ubo, vk_scene_ubo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_model_ubo, vk_model_ubo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_point_ssbo, vk_point_ssbo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_spot_ssbo, vk_spot_ssbo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_cube_vb, vk_cube_vb_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_cube_ib, vk_cube_ib_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_light_cube_vb, vk_light_cube_vb_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_light_cube_ib, vk_light_cube_ib_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_cone_vb, vk_cone_vb_alloc);

    vkDestroyCommandPool(ctx.device, g_command_pool, nullptr);
}

void update(double time) {
    static double last_time = 0.0;
    float dt = static_cast<float>(time - last_time);
    last_time = time;
    if (dt > 0.1f) dt = 0.1f;
    g_time = static_cast<float>(time);

    if (state.animate_scene) {
        for (auto& m : g_models) {
            if (!m.is_static) {
                m.rotation.y += dt * state.anim_speed * 45.0f;
                if (m.rotation.y > 180.0f) m.rotation.y -= 360.0f;
            }
        }
    }

    float speed = 3.0f * dt;
    glm::vec3 front;
    front.x = cos(glm::radians(state.camera_yaw)) * cos(glm::radians(state.camera_pitch));
    front.y = sin(glm::radians(state.camera_pitch));
    front.z = sin(glm::radians(state.camera_yaw)) * cos(glm::radians(state.camera_pitch));
    front = glm::normalize(front);
    glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0)));

    if (ImGui::IsKeyDown(ImGuiKey_W)) state.camera_pos += front * speed;
    if (ImGui::IsKeyDown(ImGuiKey_S)) state.camera_pos -= front * speed;
    if (ImGui::IsKeyDown(ImGuiKey_A)) state.camera_pos -= right * speed;
    if (ImGui::IsKeyDown(ImGuiKey_D)) state.camera_pos += right * speed;

    ImGui::Begin("Lab 3: Texture Mapping");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", glm::value_ptr(state.camera_pos), -20.0f, 20.0f);
        ImGui::SliderFloat("Yaw", &state.camera_yaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &state.camera_pitch, -89.0f, 89.0f);
        ImGui::SliderFloat("FOV", &state.fov, 10.0f, 120.0f);
    }

    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Animate", &state.animate_scene);
        ImGui::SliderFloat("Anim Speed", &state.anim_speed, 0.0f, 5.0f);
        ImGui::Checkbox("Show Light Visuals", &state.show_light_visuals);
    }

    if (ImGui::CollapsingHeader("Models", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < g_models.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            
            ImGui::Text("Model %d %s", static_cast<int>(i), g_models[i].is_static ? "(Static)" : "");
            ImGui::SliderFloat3("Pos", glm::value_ptr(g_models[i].position), -20.0f, 20.0f);
            
            
            if (!g_models[i].is_static) {
                ImGui::SliderFloat3("Rot", glm::value_ptr(g_models[i].rotation), -180.0f, 180.0f);
            }
            
            ImGui::SliderFloat3("Scale", glm::value_ptr(g_models[i].scale), 0.1f, 5.0f);
            if (ImGui::Combo("Material", &g_models[i].material_index, "Brick\0Wood\0Metal\0Red Floor\0White Wall\0")) {}
            ImGui::Checkbox("Nontrivial Sampling", &g_models[i].enable_nontrivial);
            
            if (ImGui::Button("Remove")) {
                g_models.erase(g_models.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::Separator();
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Material")) {
        ImGui::ColorEdit3("Albedo Tint", glm::value_ptr(state.albedo));
        ImGui::ColorEdit3("Specular Tint", glm::value_ptr(state.specular));
        ImGui::SliderFloat("Shininess", &state.shininess, 1.0f, 256.0f);
    }

    if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient", glm::value_ptr(state.ambient));
        ImGui::SliderFloat3("Dir Dir", glm::value_ptr(state.dir_light_dir), -1.0f, 1.0f);
        ImGui::ColorEdit3("Dir Color", glm::value_ptr(state.dir_light_color));
        ImGui::SliderFloat("Dir Intensity", &state.dir_light_intensity, 0.0f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Point Lights")) {
        if (ImGui::Button("Add Point Light")) {
            state.point_lights.push_back({{0, 2, 0, 10.0f}, {1, 1, 1, 1}});
        }
        for (size_t i = 0; i < state.point_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i) + 1000);
            ImGui::Text("Point Light %d", static_cast<int>(i));
            ImGui::SliderFloat3("Pos", glm::value_ptr(state.point_lights[i].position), -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", glm::value_ptr(state.point_lights[i].color));
            ImGui::SliderFloat("Intensity", &state.point_lights[i].position.w, 0.0f, 50.0f);
            if (ImGui::Button("Remove")) {
                state.point_lights.erase(state.point_lights.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::Separator();
            ImGui::PopID();
        }
    }

    if (ImGui::CollapsingHeader("Spot Lights")) {
        if (ImGui::Button("Add Spot Light")) {
            state.spot_lights.push_back({{0, 5, 0, 20.0f}, {0, -1, 0, 0}, {1, 1, 1, 1}, 20.0f, 40.0f});
        }
        for (size_t i = 0; i < state.spot_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i) + 2000);
            ImGui::Text("Spot Light %d", static_cast<int>(i));
            ImGui::SliderFloat3("Pos", glm::value_ptr(state.spot_lights[i].position), -10.0f, 10.0f);
            ImGui::SliderFloat3("Dir", glm::value_ptr(state.spot_lights[i].direction), -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", glm::value_ptr(state.spot_lights[i].color));
            ImGui::SliderFloat("Intensity", &state.spot_lights[i].position.w, 0.0f, 50.0f);
            ImGui::SliderFloat("Inner", &state.spot_lights[i].inner_angle, 1.0f, 89.0f);
            ImGui::SliderFloat("Outer", &state.spot_lights[i].outer_angle, 1.0f, 90.0f);
            if (ImGui::Button("Remove")) {
                state.spot_lights.erase(state.spot_lights.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::Separator();
            ImGui::PopID();
        }
    }

    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    glm::mat4 view = glm::lookAt(state.camera_pos, state.camera_pos + glm::vec3(
        cos(glm::radians(state.camera_yaw)) * cos(glm::radians(state.camera_pitch)),
        sin(glm::radians(state.camera_pitch)),
        sin(glm::radians(state.camera_yaw)) * cos(glm::radians(state.camera_pitch))
    ), glm::vec3(0, 1, 0));

    float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);
    glm::mat4 proj = glm::perspective(glm::radians(state.fov), aspect, 0.1f, 100.0f);
    proj[1][1] *= -1.0f;

    scene_ubo_mem->view_projection = proj * view;
    scene_ubo_mem->camera_position = glm::vec4(state.camera_pos, 1.0f);
    scene_ubo_mem->dir_light_direction = glm::vec4(glm::normalize(state.dir_light_dir), 1.0f);
    scene_ubo_mem->dir_light_color = glm::vec4(state.dir_light_color, state.dir_light_intensity);
    scene_ubo_mem->ambient_light = glm::vec4(state.ambient, 1.0f);
    scene_ubo_mem->point_light_count = std::min(static_cast<int>(state.point_lights.size()), 100);
    scene_ubo_mem->spot_light_count = std::min(static_cast<int>(state.spot_lights.size()), 100);

    if (scene_ubo_mem->point_light_count > 0) {
        memcpy(point_ssbo_mem, state.point_lights.data(), scene_ubo_mem->point_light_count * sizeof(PointLight));
    }
    if (scene_ubo_mem->spot_light_count > 0) {
        memcpy(spot_ssbo_mem, state.spot_lights.data(), scene_ubo_mem->spot_light_count * sizeof(SpotLight));
    }

    // Запись в dynamic UBO для всех моделей
    uint32_t model_count = std::min(static_cast<uint32_t>(g_models.size()), max_models);
    for (uint32_t i = 0; i < model_count; ++i) {
        ModelUniforms* mu = reinterpret_cast<ModelUniforms*>(vk_model_ubo_mem + i * model_ubo_aligned_size);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, g_models[i].position);
        model = glm::rotate(model, glm::radians(g_models[i].rotation.x), glm::vec3(1, 0, 0));
        model = glm::rotate(model, glm::radians(g_models[i].rotation.y), glm::vec3(0, 1, 0));
        model = glm::rotate(model, glm::radians(g_models[i].rotation.z), glm::vec3(0, 0, 1));
        model = glm::scale(model, g_models[i].scale);
        mu->model = model;
        mu->albedo = glm::vec4(state.albedo, 1.0f);
        mu->specular = glm::vec4(state.specular, 1.0f);
        mu->shininess = state.shininess;
    }

    vkResetCommandBuffer(fd.command_buffer, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(fd.command_buffer, &begin);

    VkClearValue clear_values[2];
    clear_values[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp_begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp_begin.renderPass = ctx.render_pass;
    rp_begin.framebuffer = fd.framebuffer;
    rp_begin.renderArea = {{0, 0}, ctx.swapchain_extent};
    rp_begin.clearValueCount = 2;
    rp_begin.pClearValues = clear_values;
    vkCmdBeginRenderPass(fd.command_buffer, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{0, 0, (float)ctx.swapchain_extent.width, (float)ctx.swapchain_extent.height, 0, 1};
    VkRect2D scissor{{0, 0}, ctx.swapchain_extent};
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_main_pipeline);
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_cube_vb, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, vk_cube_ib, 0, VK_INDEX_TYPE_UINT32);

    VkDescriptorSet sets[] = { vk_scene_ds, vk_model_ds, VK_NULL_HANDLE };
    for (uint32_t i = 0; i < model_count; ++i) {
        uint32_t dynamicOffset = i * static_cast<uint32_t>(model_ubo_aligned_size);
        sets[2] = g_materials[g_models[i].material_index].descriptor_set;
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout, 0, 3, sets, 1, &dynamicOffset);

        PipelineConsts pc;
        pc.enable_nontrivial = g_models[i].enable_nontrivial ? 1 : 0;
        pc.time = g_time;
        vkCmdPushConstants(fd.command_buffer, vk_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PipelineConsts), &pc);

        vkCmdDrawIndexed(fd.command_buffer, cube_index_count, 1, 0, 0, 0);
    }

    if (state.show_light_visuals) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_light_pipeline);
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_light_cube_vb, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_light_cube_ib, 0, VK_INDEX_TYPE_UINT32);

        for (const auto& pl : state.point_lights) {
            glm::mat4 lightModel = glm::translate(glm::mat4(1.0f), glm::vec3(pl.position)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
            LightPushConstants pc;
            pc.mvp = proj * view * lightModel;
            pc.color = glm::vec4(glm::vec3(pl.color), 1.0f);
            vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(LightPushConstants), &pc);
            vkCmdDrawIndexed(fd.command_buffer, 36, 1, 0, 0, 0);
        }

        for (const auto& sl : state.spot_lights) {
            glm::mat4 lightModel = glm::translate(glm::mat4(1.0f), glm::vec3(sl.position)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
            LightPushConstants pc;
            pc.mvp = proj * view * lightModel;
            pc.color = glm::vec4(glm::vec3(sl.color), 1.0f);
            vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(LightPushConstants), &pc);
            vkCmdDrawIndexed(fd.command_buffer, 36, 1, 0, 0, 0);
        }

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_line_pipeline);
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_cone_vb, offsets);

        for (const auto& sl : state.spot_lights) {
            glm::vec3 dir = glm::normalize(glm::vec3(sl.direction));
            glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(sl.position));

            glm::vec3 up(0, 1, 0);
            if (abs(glm::dot(dir, up)) > 0.99f) up = glm::vec3(0, 0, 1);

            glm::mat4 rotation = glm::lookAt(glm::vec3(0.0f), dir, up);
            model = model * glm::inverse(rotation);

            float angleRad = glm::radians(sl.outer_angle);
            float scaleXY = 4.0f * glm::tan(angleRad);
            model = glm::scale(model, glm::vec3(scaleXY, scaleXY, 1.0f));

            LightPushConstants pc;
            pc.mvp = proj * view * model;
            pc.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(LightPushConstants), &pc);
            vkCmdDraw(fd.command_buffer, 64, 1, 0, 0);
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application