#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "application.hpp"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace application {

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

const Vertex vertices[] = {
	{{-1, -1, -1}, {0.0f, 0.0f, 0.0f}},
	{{ 1, -1, -1}, {1.0f, 0.0f, 0.0f}},
	{{ 1,  1, -1}, {1.0f, 1.0f, 0.0f}},
	{{-1,  1, -1}, {0.0f, 1.0f, 0.0f}},
	{{-1, -1,  1}, {0.0f, 0.0f, 1.0f}},
	{{ 1, -1,  1}, {1.0f, 0.0f, 1.0f}},
	{{ 1,  1,  1}, {1.0f, 1.0f, 1.0f}},
	{{-1,  1,  1}, {0.0f, 1.0f, 1.0f}},
};

const uint32_t indices[] = {
	4, 5, 6,  4, 6, 7,
	1, 0, 3,  1, 3, 2,
	5, 1, 2,  5, 2, 6,
	0, 4, 7,  0, 7, 3,
	3, 7, 6,  3, 6, 2,
	1, 5, 4,  1, 4, 0,
};

struct GlobalUniforms {
	glm::mat4 mvp;
	glm::vec4 color_tint;
};

VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vk_vertex_buffer_allocation = VK_NULL_HANDLE;

VkBuffer vk_index_buffer = VK_NULL_HANDLE;
VmaAllocation vk_index_buffer_allocation = VK_NULL_HANDLE;

VkBuffer vk_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation vk_uniform_buffer_allocation = VK_NULL_HANDLE;
GlobalUniforms* vk_uniform_buffer_memory = nullptr;

VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE;
VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
VkPipeline vk_pipeline = VK_NULL_HANDLE;

struct State {
	bool use_perspective = true;
	float fov = 60.0f;
	float ortho_size = 3.0f;

	glm::vec3 position = {0.0f, 0.0f, 0.0f};
	glm::vec3 rotation = {0.0f, 0.0f, 0.0f};
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};

	bool auto_trajectory = false;
	bool animate = true;
	float animation_speed = 1.0f;
	float rotation_speed = 1.0f;
	float trajectory_radius = 2.0f;
	float trajectory_angle = 0.0f;

	glm::vec3 color_tint = {1.0f, 1.0f, 1.0f};
} state;

VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "[shader] Failed to open: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	size_t size = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(size);
	file.seekg(0);
	file.read(buffer.data(), size);
	file.close();

	VkShaderModuleCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	info.codeSize = size;
	info.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "[shader] Failed to create module: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	std::cerr << "[shader] Loaded: " << path << '\n';
	return module;
}

bool createPipeline() {
	auto& ctx = graphics::internal::context;

	VkShaderModule vert_module = loadShaderModule("shaders/shader.vert.spv");
	VkShaderModule frag_module = loadShaderModule("shaders/shader.frag.spv");
	if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) return false;

	VkPipelineShaderStageCreateInfo stages[2] = {};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vert_module;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = frag_module;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{};
	binding.binding = 0;
	binding.stride = sizeof(Vertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributes[2] = {};
	attributes[0].location = 0;
	attributes[0].binding = 0;
	attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[0].offset = offsetof(Vertex, position);
	attributes[1].location = 1;
	attributes[1].binding = 0;
	attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[1].offset = offsetof(Vertex, color);

	VkPipelineVertexInputStateCreateInfo vertex_input{};
	vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input.vertexBindingDescriptionCount = 1;
	vertex_input.pVertexBindingDescriptions = &binding;
	vertex_input.vertexAttributeDescriptionCount = 2;
	vertex_input.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo input_assembly{};
	input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewport_state{};
	viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state.viewportCount = 1;
	viewport_state.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo raster{};
	raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_BACK_BIT;
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
	blend_attachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blend_attachment.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo blend{};
	blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1;
	blend.pAttachments = &blend_attachment;

	VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo dynamic{};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dynamic_states;

	VkDescriptorSetLayoutBinding layout_binding{};
	layout_binding.binding = 0;
	layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	layout_binding.descriptorCount = 1;
	layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo dsl{};
	dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dsl.bindingCount = 1;
	dsl.pBindings = &layout_binding;

	if (vkCreateDescriptorSetLayout(ctx.device, &dsl, nullptr, &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	VkPipelineLayoutCreateInfo pl{};
	pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pl.setLayoutCount = 1;
	pl.pSetLayouts = &vk_descriptor_set_layout;

	if (vkCreatePipelineLayout(ctx.device, &pl, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

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
	pipeline_info.layout = vk_pipeline_layout;
	pipeline_info.renderPass = ctx.render_pass;
	pipeline_info.subpass = 0;

	if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info,
	                              nullptr, &vk_pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}
	std::cerr << "[app] Pipeline created\n";

	vkDestroyShaderModule(ctx.device, vert_module, nullptr);
	vkDestroyShaderModule(ctx.device, frag_module, nullptr);
	return true;
}

bool createBuffers() {
	auto& ctx = graphics::internal::context;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
	                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

	{
		VkBufferCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(vertices);
		info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationInfo ai;
		if (vmaCreateBuffer(ctx.allocator, &info, &alloc_info,
		                    &vk_vertex_buffer, &vk_vertex_buffer_allocation, &ai) != VK_SUCCESS) {
			std::cerr << "Failed to create vertex buffer\n";
			return false;
		}
		std::memcpy(ai.pMappedData, vertices, sizeof(vertices));
		std::cerr << "[app] Vertex buffer created: " << vk_vertex_buffer << '\n';
	}

	{
		VkBufferCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(indices);
		info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationInfo ai;
		if (vmaCreateBuffer(ctx.allocator, &info, &alloc_info,
		                    &vk_index_buffer, &vk_index_buffer_allocation, &ai) != VK_SUCCESS) {
			std::cerr << "Failed to create index buffer\n";
			return false;
		}
		std::memcpy(ai.pMappedData, indices, sizeof(indices));
	}

	{
		VkBufferCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(GlobalUniforms);
		info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationInfo ai;
		if (vmaCreateBuffer(ctx.allocator, &info, &alloc_info,
		                    &vk_uniform_buffer, &vk_uniform_buffer_allocation, &ai) != VK_SUCCESS) {
			std::cerr << "Failed to create uniform buffer\n";
			return false;
		}
		vk_uniform_buffer_memory = static_cast<GlobalUniforms*>(ai.pMappedData);
	}

	return true;
}

bool createDescriptors() {
	auto& ctx = graphics::internal::context;

	VkDescriptorPoolSize pool_size{};
	pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pool_size.descriptorCount = 1;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes = &pool_size;

	if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetAllocateInfo alloc{};
	alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc.descriptorPool = vk_descriptor_pool;
	alloc.descriptorSetCount = 1;
	alloc.pSetLayouts = &vk_descriptor_set_layout;

	if (vkAllocateDescriptorSets(ctx.device, &alloc, &vk_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	VkDescriptorBufferInfo buffer_info{};
	buffer_info.buffer = vk_uniform_buffer;
	buffer_info.offset = 0;
	buffer_info.range = sizeof(GlobalUniforms);

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = vk_descriptor_set;
	write.dstBinding = 0;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	write.pBufferInfo = &buffer_info;

	vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
	std::cerr << "[app] Descriptors created\n";
	return true;
}

} // namespace

bool initialize() {
	std::cerr << "[app] createBuffers\n";
	if (!createBuffers()) return false;
	std::cerr << "[app] createPipeline\n";
	if (!createPipeline()) return false;
	std::cerr << "[app] createDescriptors\n";
	if (!createDescriptors()) return false;
	std::cerr << "[app] initialize OK\n";
	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkQueueWaitIdle(ctx.graphics_queue);

	if (vk_pipeline) vkDestroyPipeline(ctx.device, vk_pipeline, nullptr);
	if (vk_pipeline_layout) vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);
	if (vk_descriptor_pool) vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
	if (vk_descriptor_set_layout) vkDestroyDescriptorSetLayout(ctx.device, vk_descriptor_set_layout, nullptr);

	if (vk_uniform_buffer) vmaDestroyBuffer(ctx.allocator, vk_uniform_buffer, vk_uniform_buffer_allocation);
	if (vk_index_buffer) vmaDestroyBuffer(ctx.allocator, vk_index_buffer, vk_index_buffer_allocation);
	if (vk_vertex_buffer) vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
}

void update(double time) {
	static double last_time = 0.0;
	double dt = time - last_time;
	last_time = time;
	if (dt > 0.1) dt = 0.1;

	if (state.animate) {
		if (state.auto_trajectory) {
			state.trajectory_angle += static_cast<float>(dt) * state.animation_speed;
		}
		state.rotation.x += static_cast<float>(dt) * state.rotation_speed * 30.0f;
		state.rotation.y += static_cast<float>(dt) * state.rotation_speed * 45.0f;
		if (state.rotation.x > 180.0f) state.rotation.x -= 360.0f;
		if (state.rotation.y > 180.0f) state.rotation.y -= 360.0f;
	}

	if (state.auto_trajectory) {
		const float a = state.trajectory_angle;
		const float r = state.trajectory_radius;

		state.position.x = (0.05f * r) * a * std::cos(a);
		state.position.y = (0.05f * r) * a * std::sin(a);
		state.position.z = 0.0f;
	}

	ImGui::Begin("Laba №1");

	ImGui::SeparatorText("Projection");
	int proj = state.use_perspective ? 0 : 1;
	if (ImGui::RadioButton("Perspective", proj == 0)) state.use_perspective = true;
	ImGui::SameLine();
	if (ImGui::RadioButton("Orthographic", proj == 1)) state.use_perspective = false;

	if (state.use_perspective) {
		ImGui::SliderFloat("size", &state.fov, 10.0f, 120.0f, "%.1f deg");
	} else {
		ImGui::SliderFloat("ortho size", &state.ortho_size, 0.5f, 10.0f);
	}

	ImGui::SeparatorText("Transform");
	if (!state.auto_trajectory) {
		ImGui::SliderFloat3("Position", glm::value_ptr(state.position), -5.0f, 5.0f);
	} else {
		ImGui::BeginDisabled();
		ImGui::SliderFloat3("Position", glm::value_ptr(state.position), -5.0f, 5.0f);
		ImGui::EndDisabled();
	}
	ImGui::SliderFloat3("Rotation", glm::value_ptr(state.rotation), -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale", glm::value_ptr(state.scale), 0.1f, 3.0f);

	ImGui::SeparatorText("Animation");
	ImGui::Checkbox("Auto trajectory", &state.auto_trajectory);
	ImGui::Checkbox("Animate", &state.animate);
	ImGui::SliderFloat("Animation speed", &state.animation_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Rotation speed", &state.rotation_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Trajectory radius", &state.trajectory_radius, 0.0f, 5.0f);

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Color tint", glm::value_ptr(state.color_tint));

	if (ImGui::Button("Reset")) {
		state = State{};
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& ctx = graphics::internal::context;

	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, state.position);
	model = glm::rotate(model, glm::radians(state.rotation.x), glm::vec3(1, 0, 0));
	model = glm::rotate(model, glm::radians(state.rotation.y), glm::vec3(0, 1, 0));
	model = glm::rotate(model, glm::radians(state.rotation.z), glm::vec3(0, 0, 1));
	model = glm::scale(model, state.scale);

	glm::mat4 view = glm::lookAt(
		glm::vec3(0.0f, 0.0f, 5.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));

	float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);
	glm::mat4 projection;
	if (state.use_perspective) {
		projection = glm::perspective(glm::radians(state.fov), aspect, 0.1f, 100.0f);
	} else {
		float h = state.ortho_size;
		float w = h * aspect;
		projection = glm::ortho(-w, w, -h, h, 0.1f, 100.0f);
	}
	projection[1][1] *= -1.0f;

	glm::mat4 mvp = projection * view * model;

	vk_uniform_buffer_memory->mvp = mvp;
	vk_uniform_buffer_memory->color_tint = glm::vec4(state.color_tint, 1.0f);

	vkResetCommandBuffer(fd.command_buffer, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &begin);

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.1f, 0.1f, 0.1f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = ctx.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = ctx.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport = {
		.x = 0, .y = 0,
		.width = float(ctx.swapchain_extent.width),
		.height = float(ctx.swapchain_extent.height),
		.minDepth = 0, .maxDepth = 1,
	};
	VkRect2D scissor = { .extent = ctx.swapchain_extent };

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_vertex_buffer, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        vk_pipeline_layout, 0, 1, &vk_descriptor_set, 0, nullptr);

	vkCmdDrawIndexed(fd.command_buffer, 36, 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application