#include "scene.h"
#include "application.h"

Scene::Scene()
{
	TexturePaths noTextures;

	/* ======================== */
	/* === World Grid Lines === */
	/* ======================== */
	std::shared_ptr<Object> grid = std::make_shared<Object>(Object(CreateGroundGrid(), noTextures, Lines));
	grid->m_DepthTest = false;
	grid->m_ModelNormalMatrix = glm::mat3(m_ClearColor, glm::vec3(0.0f), glm::vec3(0.0f)); /* We'll store the clear color in the grid's normal matrix... */
	m_RasterObjects.push_back(grid);

	std::shared_ptr<Object> axes = std::make_shared<Object>(Object(CreateXYAxes(), noTextures, Lines));
	axes->m_DepthTest = false;
	axes->m_ModelNormalMatrix = glm::mat3(m_ClearColor, glm::vec3(0.0f), glm::vec3(0.0f));
	m_RasterObjects.push_back(axes);

	/* ===================== */
	/* === Scene Objects === */
	/* ===================== */
	TexturePaths vikingRoomTextures;
	vikingRoomTextures.diffuse = "res/textures/viking_room_diff.png";
	std::shared_ptr<Object> vikingRoom = std::make_shared<Object>(Object(LoadMesh("res/meshes/viking_room.obj"), vikingRoomTextures, Flat));
	vikingRoom->Translate(0.0f, 0.0f, 0.5f);
	vikingRoom->Scale(5.0f);
	m_RasterObjects.push_back(vikingRoom);
	m_RayTraceObjects.push_back(vikingRoom);

	std::shared_ptr<Object> dragon = std::make_shared<Object>(LoadMesh("res/meshes/dragon.obj"), noTextures, Solid);
	dragon->Translate(10.0f, 0.0f, 0.0f);
	dragon->Rotate(glm::vec3(0.0f, 0.0f, 1.0f), 45.0f);
	dragon->Scale(5.0f);
	m_RasterObjects.push_back(dragon);
	m_RayTraceObjects.push_back(dragon);

	TexturePaths planeTextures;
	planeTextures.diffuse = "res/textures/white.png";
	std::shared_ptr<Object> plane0 = std::make_shared<Object>(Object(CreatePlane(), planeTextures, Flat));
	plane0->Scale(100.0f);
	m_RasterObjects.push_back(plane0);
	m_RayTraceObjects.push_back(plane0);
}


Scene::~Scene()
{
	VkCleanup();
}


void Scene::VkSetup(ImVec2 viewportSize, VkSampleCountFlagBits sampleCount, VkRenderPass& renderPass, std::vector<VkFramebuffer>& framebuffers)
{
	m_ViewportSize = viewportSize;
	m_MSAASampleCount = sampleCount;
	m_ViewportRenderPass = renderPass;
	m_ViewportFramebuffers = framebuffers;

	/* Descriptor set layout creation: uniforms, textures/samplers */
	std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

	/* We'll have 1 ubo to pass in mesh data like its model & normal matrices */
	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.pImmutableSamplers = nullptr;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	layoutBindings.push_back(uboLayoutBinding);

	/* We'll have 3 samplers for diffuse, specular, and normal textures */
	VkDescriptorSetLayoutBinding diffuseSamplerLayoutBinding{};
	diffuseSamplerLayoutBinding.binding = 1;
	diffuseSamplerLayoutBinding.descriptorCount = 1;
	diffuseSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	diffuseSamplerLayoutBinding.pImmutableSamplers = nullptr;
	diffuseSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	layoutBindings.push_back(diffuseSamplerLayoutBinding);

	VkDescriptorSetLayoutBinding specularSamplerLayoutBinding{};
	specularSamplerLayoutBinding.binding = 2;
	specularSamplerLayoutBinding.descriptorCount = 1;
	specularSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	specularSamplerLayoutBinding.pImmutableSamplers = nullptr;
	specularSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	layoutBindings.push_back(specularSamplerLayoutBinding);

	VkDescriptorSetLayoutBinding normalSamplerLayoutBinding{};
	normalSamplerLayoutBinding.binding = 3;
	normalSamplerLayoutBinding.descriptorCount = 1;
	normalSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	normalSamplerLayoutBinding.pImmutableSamplers = nullptr;
	normalSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	layoutBindings.push_back(normalSamplerLayoutBinding);

	vk::CreateDescriptorSetLayout(layoutBindings, m_DescriptorSetLayout);
	vk::CreateDescriptorPool(1000, m_DescriptorPool); /* Note: Make sure to update the max number of descriptor sets according to the number of objects you have! */

	/* Generate graphics pipelines with different shaders */
	PipelineInfo pInfo;
	pInfo.descriptorPool = m_DescriptorPool;
	pInfo.descriptorSetLayout = m_DescriptorSetLayout;

	std::vector<std::string> shadersFlat = { "src/vulkan/shaders/Flat.vert", "src/vulkan/shaders/Flat.frag" };
	pInfo.pipeline = vk::CreateGraphicsPipeline(shadersFlat, m_ViewportSize, m_MSAASampleCount, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, m_ViewportRenderPass, m_DescriptorSetLayout, m_PipelineLayout);
	pInfo.pipelineLayout = m_PipelineLayout; /* Note: has to be after pipeline creation bc pipeline layout is created in CreateGraphicsPipeline() */
	m_Pipelines[Flat] = pInfo;

	std::vector<std::string> shadersSolid = { "src/vulkan/shaders/Solid.vert", "src/vulkan/shaders/Solid.frag" };
	pInfo.pipeline = vk::CreateGraphicsPipeline(shadersSolid, m_ViewportSize, m_MSAASampleCount, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, m_ViewportRenderPass, m_DescriptorSetLayout, m_PipelineLayout);
	m_Pipelines[Solid] = pInfo;

	std::vector<std::string> shadersNormal = { "src/vulkan/shaders/Solid.vert", "src/vulkan/shaders/Normal.frag" };
	pInfo.pipeline = vk::CreateGraphicsPipeline(shadersNormal, m_ViewportSize, m_MSAASampleCount, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, m_ViewportRenderPass, m_DescriptorSetLayout, m_PipelineLayout);
	m_Pipelines[Normal] = pInfo;

	std::vector<std::string> shadersLines = { "src/vulkan/shaders/Lines.vert", "src/vulkan/shaders/Lines.frag" };
	pInfo.pipeline = vk::CreateGraphicsPipeline(shadersLines, m_ViewportSize, m_MSAASampleCount, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, m_ViewportRenderPass, m_DescriptorSetLayout, m_PipelineLayout);
	m_Pipelines[Lines] = pInfo;

	
	/* Setup objects for rendering with Vulkan */
	for (auto& obj : m_RasterObjects)
	{
		obj->VkSetup(m_Pipelines[static_cast<PipelineType>(obj->m_PipelineType)]);
		obj->VkUpdateUniformBuffer();
	}
}


void Scene::VkResize(ImVec2 newSize, std::vector<VkFramebuffer>& framebuffers)
{
	m_ViewportSize = newSize;
	m_ViewportFramebuffers = framebuffers;
}


void Scene::VkDraw(const Camera& camera)
{
	VkCommandBuffer commandBuffer = vk::GetGraphicsCommandBuffer();


	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = m_ViewportRenderPass;
	renderPassInfo.framebuffer = m_ViewportFramebuffers[vk::MainWindowData.FrameIndex];
	renderPassInfo.renderArea.offset = { 0, 0 };
	renderPassInfo.renderArea.extent = { static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y) };

	std::array<VkClearValue, 2> clearValues{};
	clearValues[0].color = { {m_ClearColor.x, m_ClearColor.y, m_ClearColor.z, 1.0f} };
	clearValues[1].depthStencil = { 1.0f, 0 };
	renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderPassInfo.pClearValues = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	/* Need to set the viewport and scissor since they are dynamic */
	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = m_ViewportSize.x;
	viewport.height = m_ViewportSize.y;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = { static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y) };
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	/* Set push constants */
	PushConstants constants;
	constants.view = camera.view_matrix;
	constants.proj = camera.projection_matrix;
	vkCmdPushConstants(commandBuffer, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &constants);

	/* Draw the objects */
	for (auto object : m_RasterObjects)
	{
		object->VkDraw(commandBuffer);
	}

	vkCmdEndRenderPass(commandBuffer);


	vk::FlushGraphicsCommandBuffer(commandBuffer);
}


void Scene::VkCleanup()
{
	vkDestroyDescriptorPool(vk::Device, m_DescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(vk::Device, m_DescriptorSetLayout, nullptr);

	auto it = m_Pipelines.begin();
	while (it != m_Pipelines.end())
	{
		vkDestroyPipeline(vk::Device, it->second.pipeline, nullptr);
	}

	vkDestroyPipelineLayout(vk::Device, m_PipelineLayout, nullptr);
}