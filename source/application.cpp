#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "application.hpp"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

namespace {

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
};

void generateCube(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
    vertices.push_back({{-1, -1,  1}, { 0,  0,  1}});
    vertices.push_back({{ 1, -1,  1}, { 0,  0,  1}});
    vertices.push_back({{ 1,  1,  1}, { 0,  0,  1}});
    vertices.push_back({{-1,  1,  1}, { 0,  0,  1}});

    vertices.push_back({{ 1, -1, -1}, { 0,  0, -1}});
    vertices.push_back({{-1, -1, -1}, { 0,  0, -1}});
    vertices.push_back({{-1,  1, -1}, { 0,  0, -1}});
    vertices.push_back({{ 1,  1, -1}, { 0,  0, -1}});

    vertices.push_back({{-1,  1,  1}, { 0,  1,  0}});
    vertices.push_back({{ 1,  1,  1}, { 0,  1,  0}});
    vertices.push_back({{ 1,  1, -1}, { 0,  1,  0}});
    vertices.push_back({{-1,  1, -1}, { 0,  1,  0}});

    vertices.push_back({{-1, -1, -1}, { 0, -1,  0}});
    vertices.push_back({{ 1, -1, -1}, { 0, -1,  0}});
    vertices.push_back({{ 1, -1,  1}, { 0, -1,  0}});
    vertices.push_back({{-1, -1,  1}, { 0, -1,  0}});

    vertices.push_back({{ 1, -1,  1}, { 1,  0,  0}});
    vertices.push_back({{ 1, -1, -1}, { 1,  0,  0}});
    vertices.push_back({{ 1,  1, -1}, { 1,  0,  0}});
    vertices.push_back({{ 1,  1,  1}, { 1,  0,  0}});

    vertices.push_back({{-1, -1, -1}, {-1,  0,  0}});
    vertices.push_back({{-1, -1,  1}, {-1,  0,  0}});
    vertices.push_back({{-1,  1,  1}, {-1,  0,  0}});
    vertices.push_back({{-1,  1, -1}, {-1,  0,  0}});

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

        vertices.push_back({tip, {0,0,0}});
        vertices.push_back({base, {0,0,0}});
    }

    for (int i = 0; i < segments; ++i) {
        float a1 = (float)i / segments * 2.0f * 3.14159265f;
        float a2 = (float)(i + 1) / segments * 2.0f * 3.14159265f;
        
        glm::vec3 p1(radius * cos(a1), radius * sin(a1), -height);
        glm::vec3 p2(radius * cos(a2), radius * sin(a2), -height);

        vertices.push_back({p1, {0,0,0}});
        vertices.push_back({p2, {0,0,0}});
    }
}

std::vector<Vertex> cubeVertices;
std::vector<uint32_t> cubeIndices;

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

struct PushConstants {
    glm::mat4 mvp;
    glm::vec4 color;
};

VkBuffer vk_vertex_buffer, vk_index_buffer;
VmaAllocation vk_vertex_alloc, vk_index_alloc;

VkBuffer vk_scene_ubo, vk_model_ubo;
VmaAllocation vk_scene_ubo_alloc, vk_model_ubo_alloc;
SceneUniforms* scene_ubo_mem = nullptr;
ModelUniforms* model_ubo_mem = nullptr;

VkBuffer vk_point_ssbo, vk_spot_ssbo;
VmaAllocation vk_point_ssbo_alloc, vk_spot_ssbo_alloc;
void* point_ssbo_mem = nullptr;
void* spot_ssbo_mem = nullptr;

VkBuffer vk_cone_vb;
VmaAllocation vk_cone_vb_alloc;
VkPipeline vk_line_pipeline;

VkBuffer vk_light_cube_vb, vk_light_cube_ib;
VmaAllocation vk_light_cube_vb_alloc, vk_light_cube_ib_alloc;

VkDescriptorSetLayout vk_scene_dsl, vk_model_dsl;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_ds, vk_model_ds;
VkPipelineLayout vk_pipeline_layout, vk_light_pipeline_layout;
VkPipeline vk_main_pipeline, vk_light_pipeline;

struct AppState {
    glm::vec3 camera_pos = {3.0f, 3.0f, 5.0f};
    float camera_yaw = -135.0f;
    float camera_pitch = -30.0f;
    float fov = 60.0f;

    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
    glm::vec3 scale = {1.0f, 1.0f, 1.0f};
    bool animate = true;
    float rotation_speed = 1.0f;

    glm::vec3 albedo = {0.8f, 0.2f, 0.2f};
    glm::vec3 specular = {1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;

    glm::vec3 ambient = {0.1f, 0.1f, 0.1f};
    glm::vec3 dir_light_dir = {-0.5f, -1.0f, -0.3f};
    glm::vec3 dir_light_color = {1.0f, 1.0f, 1.0f};
    float dir_light_intensity = 1.0f;

    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;

    bool show_light_visuals = true;
} state;

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

    VkVertexInputAttributeDescription attributes[2] = {};
    attributes[0].location = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = offsetof(Vertex, position);
    attributes[1].location = 1;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = offsetof(Vertex, normal);

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = isMain ? 2 : 1;
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
    VmaAllocationCreateInfo alloc_info{};
    alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

    generateCube(cubeVertices, cubeIndices);

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = cubeVertices.size() * sizeof(Vertex);
    info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationInfo ai;
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_vertex_buffer, &vk_vertex_alloc, &ai);
    memcpy(ai.pMappedData, cubeVertices.data(), info.size);

    info.size = cubeIndices.size() * sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_index_buffer, &vk_index_alloc, &ai);
    memcpy(ai.pMappedData, cubeIndices.data(), info.size);

    info.size = sizeof(SceneUniforms);
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_scene_ubo, &vk_scene_ubo_alloc, &ai);
    scene_ubo_mem = static_cast<SceneUniforms*>(ai.pMappedData);

    info.size = sizeof(ModelUniforms);
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_model_ubo, &vk_model_ubo_alloc, &ai);
    model_ubo_mem = static_cast<ModelUniforms*>(ai.pMappedData);

    info.size = 100 * sizeof(PointLight);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_point_ssbo, &vk_point_ssbo_alloc, &ai);
    point_ssbo_mem = ai.pMappedData;

    info.size = 100 * sizeof(SpotLight);
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_spot_ssbo, &vk_spot_ssbo_alloc, &ai);
    spot_ssbo_mem = ai.pMappedData;

    std::vector<Vertex> lightCubeVerts = {
        {{-1, -1, -1}, {0,0,0}}, {{ 1, -1, -1}, {0,0,0}}, {{ 1,  1, -1}, {0,0,0}}, {{-1,  1, -1}, {0,0,0}},
        {{-1, -1,  1}, {0,0,0}}, {{ 1, -1,  1}, {0,0,0}}, {{ 1,  1,  1}, {0,0,0}}, {{-1,  1,  1}, {0,0,0}}
    };
    std::vector<uint32_t> lightCubeIdx = {
        4, 5, 6,  4, 6, 7,  1, 0, 3,  1, 3, 2,  5, 1, 2,  5, 2, 6,
        0, 4, 7,  0, 7, 3,  3, 7, 6,  3, 6, 2,  1, 5, 4,  1, 4, 0
    };
    info.size = lightCubeVerts.size() * sizeof(Vertex);
    info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_light_cube_vb, &vk_light_cube_vb_alloc, &ai);
    memcpy(ai.pMappedData, lightCubeVerts.data(), info.size);

    info.size = lightCubeIdx.size() * sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_light_cube_ib, &vk_light_cube_ib_alloc, &ai);
    memcpy(ai.pMappedData, lightCubeIdx.data(), info.size);

    std::vector<Vertex> coneVerts;
    generateSpotCone(coneVerts);
    info.size = coneVerts.size() * sizeof(Vertex);
    info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vmaCreateBuffer(ctx.allocator, &info, &alloc_info, &vk_cone_vb, &vk_cone_vb_alloc, &ai);
    memcpy(ai.pMappedData, coneVerts.data(), info.size);

    return true;
}

bool createDescriptorsAndPipelines() {
    auto& ctx = graphics::internal::context;

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

    VkDescriptorSetLayoutBinding model_bindings[1] = {};
    model_bindings[0].binding = 0;
    model_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    model_bindings[0].descriptorCount = 1;
    model_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_model{};
    dsl_model.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_model.bindingCount = 1;
    dsl_model.pBindings = model_bindings;
    vkCreateDescriptorSetLayout(ctx.device, &dsl_model, nullptr, &vk_model_dsl);

    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }
    };
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 2;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &vk_descriptor_pool);

    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = vk_descriptor_pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &vk_scene_dsl;
    vkAllocateDescriptorSets(ctx.device, &alloc, &vk_scene_ds);

    alloc.pSetLayouts = &vk_model_dsl;
    vkAllocateDescriptorSets(ctx.device, &alloc, &vk_model_ds);

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
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[3].descriptorCount = 1; writes[3].pBufferInfo = &model_ubo_info;

    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);

    VkDescriptorSetLayout layouts[] = { vk_scene_dsl, vk_model_dsl };
    VkPipelineLayoutCreateInfo pl_info{};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 2;
    pl_info.pSetLayouts = layouts;
    vkCreatePipelineLayout(ctx.device, &pl_info, nullptr, &vk_pipeline_layout);

    vk_main_pipeline = createPipeline("shaders/main.vert.spv", "shaders/main.frag.spv", vk_pipeline_layout, true);

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo light_pl_info{};
    light_pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    light_pl_info.pushConstantRangeCount = 1;
    light_pl_info.pPushConstantRanges = &push_range;
    vkCreatePipelineLayout(ctx.device, &light_pl_info, nullptr, &vk_light_pipeline_layout);

    vk_light_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", vk_light_pipeline_layout, false);
    vk_line_pipeline = createPipeline("shaders/unlit.vert.spv", "shaders/unlit.frag.spv", vk_light_pipeline_layout, false, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

    return true;
}

} 

bool initialize() {
    if (!createBuffers()) return false;
    if (!createDescriptorsAndPipelines()) return false;

    state.point_lights.push_back({{2.0f, 1.0f, 2.0f, 15.0f}, {1.0f, 0.2f, 0.2f, 1.0f}});
    state.point_lights.push_back({{-2.0f, 1.0f, 2.0f, 15.0f}, {0.2f, 0.2f, 1.0f, 1.0f}});
    state.spot_lights.push_back({{0.0f, 3.0f, 0.0f, 30.0f}, {0.0f, -1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, 15.0f, 30.0f});

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkDeviceWaitIdle(ctx.device);

    vkDestroyPipeline(ctx.device, vk_main_pipeline, nullptr);
    vkDestroyPipeline(ctx.device, vk_light_pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(ctx.device, vk_light_pipeline_layout, nullptr);
	vkDestroyPipeline(ctx.device, vk_line_pipeline, nullptr);
    vmaDestroyBuffer(ctx.allocator, vk_cone_vb, vk_cone_vb_alloc);

    vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_scene_dsl, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, vk_model_dsl, nullptr);

    vmaDestroyBuffer(ctx.allocator, vk_scene_ubo, vk_scene_ubo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_model_ubo, vk_model_ubo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_point_ssbo, vk_point_ssbo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_spot_ssbo, vk_spot_ssbo_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer, vk_vertex_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_index_buffer, vk_index_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_light_cube_vb, vk_light_cube_vb_alloc);
    vmaDestroyBuffer(ctx.allocator, vk_light_cube_ib, vk_light_cube_ib_alloc);
}

void update(double time) {
    static double last_time = 0.0;
    float dt = static_cast<float>(time - last_time);
    last_time = time;
    if (dt > 0.1f) dt = 0.1f;

    if (state.animate) {
        state.rotation.x += dt * state.rotation_speed * 30.0f;
        state.rotation.y += dt * state.rotation_speed * 45.0f;
        if (state.rotation.x > 180.0f) state.rotation.x -= 360.0f;
        if (state.rotation.y > 180.0f) state.rotation.y -= 360.0f;
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

    ImGui::Begin("Lab 2: Camera & Lighting");

    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Position", glm::value_ptr(state.camera_pos), -10.0f, 10.0f);
        ImGui::SliderFloat("Yaw", &state.camera_yaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &state.camera_pitch, -89.0f, 89.0f);
        ImGui::SliderFloat("FOV", &state.fov, 10.0f, 120.0f);
    }

    if (ImGui::CollapsingHeader("Model Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Animate Rotation", &state.animate);
        ImGui::SliderFloat("Rotation Speed", &state.rotation_speed, 0.0f, 5.0f);
        ImGui::SliderFloat3("Rotation", glm::value_ptr(state.rotation), -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale", glm::value_ptr(state.scale), 0.1f, 3.0f);
    }

    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Albedo", glm::value_ptr(state.albedo));
        ImGui::ColorEdit3("Specular", glm::value_ptr(state.specular));
        ImGui::SliderFloat("Shininess", &state.shininess, 1.0f, 256.0f);
    }

    if (ImGui::CollapsingHeader("Scene Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient Light", glm::value_ptr(state.ambient));
        ImGui::SliderFloat3("Dir Light Dir", glm::value_ptr(state.dir_light_dir), -1.0f, 1.0f);
        ImGui::ColorEdit3("Dir Light Color", glm::value_ptr(state.dir_light_color));
        ImGui::SliderFloat("Dir Intensity", &state.dir_light_intensity, 0.0f, 5.0f);
        ImGui::Checkbox("Show Light Visuals", &state.show_light_visuals);
    }

    if (ImGui::CollapsingHeader("Point Lights")) {
        if (ImGui::Button("Add Point Light")) {
            state.point_lights.push_back({{0, 2, 0, 10.0f}, {1, 1, 1, 1}});
        }
        for (size_t i = 0; i < state.point_lights.size(); ++i) {
            ImGui::PushID(static_cast<void*>(&state.point_lights[i]));
            
            ImGui::Text("Point Light %d", static_cast<int>(i));
            ImGui::SliderFloat3("Pos", glm::value_ptr(state.point_lights[i].position), -10.0f, 10.0f);
            ImGui::ColorEdit3("Color", glm::value_ptr(state.point_lights[i].color));
            ImGui::SliderFloat("Intensity", &state.point_lights[i].position.w, 0.0f, 50.0f);
            
            if (ImGui::Button(("Remove##Point" + std::to_string(i)).c_str())) {
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
            ImGui::PushID(static_cast<void*>(&state.spot_lights[i]));
            
            ImGui::Text("Spot Light %d", static_cast<int>(i));
            ImGui::SliderFloat3("Pos", glm::value_ptr(state.spot_lights[i].position), -10.0f, 10.0f);
            ImGui::SliderFloat3("Dir", glm::value_ptr(state.spot_lights[i].direction), -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", glm::value_ptr(state.spot_lights[i].color));
            ImGui::SliderFloat("Intensity", &state.spot_lights[i].position.w, 0.0f, 50.0f);
            ImGui::SliderFloat("Inner Angle", &state.spot_lights[i].inner_angle, 1.0f, 89.0f);
            ImGui::SliderFloat("Outer Angle", &state.spot_lights[i].outer_angle, 1.0f, 90.0f);
            
            if (ImGui::Button(("Remove##Spot" + std::to_string(i)).c_str())) {
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

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, state.position);
    model = glm::rotate(model, glm::radians(state.rotation.x), glm::vec3(1, 0, 0));
    model = glm::rotate(model, glm::radians(state.rotation.y), glm::vec3(0, 1, 0));
    model = glm::rotate(model, glm::radians(state.rotation.z), glm::vec3(0, 0, 1));
    model = glm::scale(model, state.scale);

    scene_ubo_mem->view_projection = proj * view;
    scene_ubo_mem->camera_position = glm::vec4(state.camera_pos, 1.0f);
    scene_ubo_mem->dir_light_direction = glm::vec4(glm::normalize(state.dir_light_dir), 1.0f);
    scene_ubo_mem->dir_light_color = glm::vec4(state.dir_light_color, state.dir_light_intensity);
    scene_ubo_mem->ambient_light = glm::vec4(state.ambient, 1.0f);
    scene_ubo_mem->point_light_count = std::min(static_cast<int>(state.point_lights.size()), 100);
    scene_ubo_mem->spot_light_count = std::min(static_cast<int>(state.spot_lights.size()), 100);

    model_ubo_mem->model = model;
    model_ubo_mem->albedo = glm::vec4(state.albedo, 1.0f);
    model_ubo_mem->specular = glm::vec4(state.specular, 1.0f);
    model_ubo_mem->shininess = state.shininess;

    if (scene_ubo_mem->point_light_count > 0) {
        memcpy(point_ssbo_mem, state.point_lights.data(), scene_ubo_mem->point_light_count * sizeof(PointLight));
    }
    if (scene_ubo_mem->spot_light_count > 0) {
        memcpy(spot_ssbo_mem, state.spot_lights.data(), scene_ubo_mem->spot_light_count * sizeof(SpotLight));
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
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, offsets);
    vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
    
    VkDescriptorSet sets[] = {vk_scene_ds, vk_model_ds};
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout, 0, 2, sets, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, static_cast<uint32_t>(cubeIndices.size()), 1, 0, 0, 0);

    if (state.show_light_visuals) {
        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_light_pipeline);
        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_light_cube_vb, offsets);
        vkCmdBindIndexBuffer(fd.command_buffer, vk_light_cube_ib, 0, VK_INDEX_TYPE_UINT32);

        for (const auto& pl : state.point_lights) {
            glm::mat4 lightModel = glm::translate(glm::mat4(1.0f), glm::vec3(pl.position)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
            PushConstants pc;
            pc.mvp = proj * view * lightModel;
            pc.color = glm::vec4(glm::vec3(pl.color), 1.0f);
            vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(fd.command_buffer, 36, 1, 0, 0, 0);
        }

        for (const auto& sl : state.spot_lights) {
            glm::mat4 lightModel = glm::translate(glm::mat4(1.0f), glm::vec3(sl.position)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
            PushConstants pc;
            pc.mvp = proj * view * lightModel;
            pc.color = glm::vec4(glm::vec3(sl.color), 1.0f);
            vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
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

            PushConstants pc;
            pc.mvp = proj * view * model;
            pc.color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            
            vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pc);
            vkCmdDraw(fd.command_buffer, 64, 1, 0, 0); 
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} //namespace application