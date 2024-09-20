#include "raster_view.h"

#include <stdlib.h>
#include <fstream>


/* ========================= */
/* === Utility functions === */
/* ========================= */

static uint32_t GetVulkanMemoryType(VkMemoryPropertyFlags properties, uint32_t type_bits)
{
	VkPhysicalDeviceMemoryProperties prop;
	vkGetPhysicalDeviceMemoryProperties(VK::PhysicalDevice, &prop);
	for (uint32_t i = 0; i < prop.memoryTypeCount; i++)
	{
		if ((prop.memoryTypes[i].propertyFlags & properties) == properties && type_bits & (1 << i))
		{
			return i;
		}
	}

	return 0xffffffff;
}


static std::vector<char> ReadFile(const std::string& filename) /* change to "ReadShaderFile" ? */
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		std::cerr << "Failed to open file: " << filename << std::endl;
		abort();
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();
	return buffer;
}


/* ================================ */
/* === RasterView Class Methods === */
/* ================================ */

/* === Standard layer methods === */

void RasterView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();

	m_PhysicalDevice = VK::PhysicalDevice;
	m_Device = VK::Device;
	//m_MinImageCount = app->GetMinImageCount();
	m_MinImageCount = 1; /* We are only rendering to texture so we don't need the whole swapchain setup */

	InitVulkan();

	m_Camera = new Camera(100, 100, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 1.0), 45.0f);

	m_Image = std::make_shared<Image>("./res/textures/teapot_normal.png");
}


void RasterView::OnDetach()
{
	CleanupVulkan();
}


void RasterView::OnUpdate()
{
	if (m_ViewportFocused)
	{
		m_Camera->Inputs(m_WindowHandle);
	}

	VkCommandBuffer commandBuffer = m_AppHandle->GetCommandBuffer();
	RecordCommandBuffer(commandBuffer, 0);
	m_AppHandle->FlushCommandBuffer(commandBuffer);
	m_DescriptorSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(m_Sampler, m_ViewportImageViews[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}


void RasterView::OnUIRender()
{
	/* No padding on viewports */
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	{
		ImGui::Begin("Rasterized Viewport");
		{
			ImGui::BeginChild("Rasterized");
			{
				m_ViewportFocused = ImGui::IsWindowFocused();
				ImVec2 tempSize = ImGui::GetContentRegionAvail();
				if (m_ViewportSize.x != tempSize.x || m_ViewportSize.y != tempSize.y)
				{
					OnResize(tempSize);
				}

				//ImGui::Image(m_Image->GetDescriptorSet(), { (float)m_Image->GetWidth(), (float)m_Image->GetHeight() });
				ImGui::Image(m_DescriptorSet, m_ViewportSize);
			}
			ImGui::EndChild();
		}
		ImGui::End();
	}
	/* Add back in padding for non-viewport ImGui */
	ImGui::PopStyleVar();

	ImGui::ShowDemoWindow();

	ImGui::Begin("Raster Debug Panel");
	{
		CommonDebug(m_ViewportSize, m_Camera);
	}
	ImGui::End();
}


/* === RasterView specific methods === */

void RasterView::InitVulkan()
{
	CreateViewportImages();
	CreateViewportImageViews();
	CreateRenderPass();
	CreateGraphicsPipeline();
	CreateFrameBuffers();
	CreateSampler();
	//CreateCommandPool();
	//CreateCommandBuffer();
}


void RasterView::CleanupVulkan()
{
	vkDestroySampler(m_Device, m_Sampler, nullptr);

	//vkDestroyCommandPool(m_Device, m_ViewportCommandPool, nullptr);

	for (auto framebuffer : m_ViewportFramebuffers)
	{
		vkDestroyFramebuffer(m_Device, framebuffer, nullptr);
	}

	vkDestroyPipeline(m_Device, m_ViewportGraphicsPipeline, nullptr);
	vkDestroyPipelineLayout(m_Device, m_ViewportPipelineLayout, nullptr);
	vkDestroyRenderPass(m_Device, m_ViewportRenderPass, nullptr);

	for (auto imageView : m_ViewportImageViews)
	{
		vkDestroyImageView(m_Device, imageView, nullptr);
	}

	for (auto image : m_ViewportImages)
	{
		vkDestroyImage(m_Device, image, nullptr);
	}
}


void RasterView::OnResize(ImVec2 newSize)
{
	m_ViewportSize = newSize;

	vkDeviceWaitIdle(m_Device);
	//for (auto framebuffer : m_ViewportFramebuffers)
	//{
	//	vkDestroyFramebuffer(m_Device, framebuffer, nullptr);
	//}
	//for (auto imageView : m_ViewportImageViews)
	//{
	//	vkDestroyImageView(m_Device, imageView, nullptr);
	//}
	//for (auto image : m_ViewportImages)
	//{
	//	vkDestroyImage(m_Device, image, nullptr);
	//}
	CreateViewportImages();
	CreateViewportImageViews();
	CreateFrameBuffers();
}


void RasterView::CreateSampler()
{
	VkSamplerCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	info.magFilter = VK_FILTER_LINEAR;
	info.minFilter = VK_FILTER_LINEAR;
	info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	info.minLod = -1000;
	info.maxLod = 1000;
	info.maxAnisotropy = 1.0f;
	VkResult err = vkCreateSampler(m_Device, &info, nullptr, &m_Sampler);
	VK::check_vk_result(err);
}


RasterView::QueueFamilyIndices RasterView::FindQueueFamilies(VkPhysicalDevice device)
{
	RasterView::QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.graphicsFamily = i;
		}

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, VK::MainWindowData.Surface, &presentSupport);

		if (presentSupport) {
			indices.presentFamily = i;
		}

		if (indices.isComplete()) {
			break;
		}

		i++;
	}

	return indices;
}


void RasterView::CreateViewportImages()
{
	VkResult err;

	m_ViewportImages.resize(m_MinImageCount);
	m_ImageDeviceMemory.resize(m_MinImageCount);

	for (uint32_t i = 0; i < m_MinImageCount; i++)
	{
		VkImageCreateInfo imageCreateInfo{};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = VK::MainWindowData.SurfaceFormat.format;
		imageCreateInfo.extent.width = (uint32_t)m_ViewportSize.x;
		imageCreateInfo.extent.height = (uint32_t)m_ViewportSize.y;
		imageCreateInfo.extent.depth = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL; /* Setting to LINEAR causes an error? */
		imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		err = vkCreateImage(m_Device, &imageCreateInfo, nullptr, &m_ViewportImages[i]);
		VK::check_vk_result(err);

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(m_Device, m_ViewportImages[i], &memRequirements);

		VkMemoryAllocateInfo memAllocInfo{};
		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllocInfo.allocationSize = memRequirements.size;
		memAllocInfo.memoryTypeIndex = GetVulkanMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements.memoryTypeBits);
		err = vkAllocateMemory(m_Device, &memAllocInfo, nullptr, &m_ImageDeviceMemory[i]);
		VK::check_vk_result(err);

		err = vkBindImageMemory(m_Device, m_ViewportImages[i], m_ImageDeviceMemory[i], 0);
		VK::check_vk_result(err);
	}
}


void RasterView::CreateViewportImageViews()
{
	VkResult err;

	m_ViewportImageViews.resize(m_MinImageCount);

	for (uint32_t i = 0; i < m_MinImageCount; i++)
	{
		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.image = m_ViewportImages[i];
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = VK::MainWindowData.SurfaceFormat.format;
		imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.layerCount = 1;
		err = vkCreateImageView(m_Device, &imageViewCreateInfo, nullptr, &m_ViewportImageViews[i]);
		VK::check_vk_result(err);
	}
}


void RasterView::CreateRenderPass()
{
	VkResult err;

	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = VK::MainWindowData.SurfaceFormat.format;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; /* we clear the color attachment with constants */
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0; /* index to attachment, e.g. "layout(location = 0) out vec4 outColor;" in frag shader code */
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	err = vkCreateRenderPass(m_Device, &renderPassInfo, nullptr, &m_ViewportRenderPass);
	VK::check_vk_result(err);
}


VkShaderModule RasterView::CreateShaderModule(const std::vector<char>& code)
{
	VkResult err;

	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	err = vkCreateShaderModule(m_Device, &createInfo, nullptr, &shaderModule);
	VK::check_vk_result(err);

	return shaderModule;
}


void RasterView::CreateGraphicsPipeline()
{
	VkResult err;

	/* ====== Shader Modules and Shader Stages ====== */
	auto vertShaderCode = ReadFile("res/shaders/spv/HelloTriangle.vert.spv");
	auto fragShaderCode = ReadFile("res/shaders/spv/HelloTriangle.frag.spv");

	VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main"; /* i.e., entry point */

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };


	/* ====== Fixed Function Stages ====== */

	/* === Vertex Input === */
	/* This is where we state the bindings and attribute layout of input data */
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.pVertexBindingDescriptions = nullptr; /* optional */
	vertexInputInfo.vertexAttributeDescriptionCount = 0;
	vertexInputInfo.pVertexAttributeDescriptions = nullptr; /* optional -- we'll set this later when drawing with actual vertex data */

	/* === Input Assembly === */
	/* Where we define the type of primitive to draw (e.g. LINE_STRIP/TRIANGLE_LIST) */
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE; /* If true and using element (index) buffers, can use special index (e.g. 0xFFFF) to restart _STRIP topology */


	/* === Viewports and Scissors === */
	/* Viewport describes the region the frame will be rendered to. Scissor defines region where pixels are actually stored. */
	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = m_ViewportSize.x;
	viewport.height = m_ViewportSize.y;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = { (uint32_t)m_ViewportSize.x, (uint32_t)m_ViewportSize.y };
	
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;
	
	/* === Rasterizer === */
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	/* === Multisampling === */
	/* We'll get back to this later? For now, disabled. */
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f; /* optional */
	multisampling.pSampleMask = nullptr; /* optional */
	multisampling.alphaToCoverageEnable = VK_FALSE; /* optional */
	multisampling.alphaToOneEnable = VK_FALSE; /* optional */

	/* === Depth and stencil testing === */
	// TODO

	/* === Color Blending === */
	/* 
	 * VkPipelineColorBlendAttachmentState = per attached framebuffer, 
	 * VkPipelineColorBlendStateCreateInfo = *global* color blending settings 
	 */
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_AND;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f; /* optional */
	colorBlending.blendConstants[1] = 0.0f; /* optional */
	colorBlending.blendConstants[2] = 0.0f; /* optional */
	colorBlending.blendConstants[3] = 0.0f; /* optional */

	/* === Dynamic States === */
	/* Since we are using them, these must be set before we draw! */
	std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();


	/* ====== Pipeline layout ====== */
	/* what we use to determine uniforms being sent to the shaders */
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 0; /* optional */
	pipelineLayoutInfo.pSetLayouts = nullptr; /* optional */
	pipelineLayoutInfo.pushConstantRangeCount = 0; /* optional */
	pipelineLayoutInfo.pPushConstantRanges = nullptr; /* optional */
	err = vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ViewportPipelineLayout);
	VK::check_vk_result(err);


	/* ====== Pipeline creation ====== */
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = nullptr; /* optional */
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = m_ViewportPipelineLayout;
	pipelineInfo.renderPass = m_ViewportRenderPass;
	pipelineInfo.subpass = 0; /* index of the subpass where this graphics pipeline will be used */
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; /* optional -- used if you are creating derivative pipelines */
	pipelineInfo.basePipelineIndex = -1; /* optional */
	err = vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_ViewportGraphicsPipeline);
	VK::check_vk_result(err);

	/* === Cleanup === */
	vkDestroyShaderModule(m_Device, vertShaderModule, nullptr);
	vkDestroyShaderModule(m_Device, fragShaderModule, nullptr);
}


void RasterView::CreateFrameBuffers()
{
	VkResult err;

	m_ViewportFramebuffers.resize(m_ViewportImageViews.size());

	for (size_t i = 0; i < m_ViewportImageViews.size(); i++)
	{
		VkImageView attachments[] = {
			m_ViewportImageViews[i]
		};

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = m_ViewportRenderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = (uint32_t)m_ViewportSize.x;
		framebufferInfo.height = (uint32_t)m_ViewportSize.y;
		framebufferInfo.layers = 1;
		err = vkCreateFramebuffer(m_Device, &framebufferInfo, nullptr, &m_ViewportFramebuffers[i]);
		VK::check_vk_result(err);
	}
}


//void RasterView::CreateCommandPool()
//{
//	VkResult err;
//
//	QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(m_PhysicalDevice);
//
//	VkCommandPoolCreateInfo poolInfo{};
//	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
//	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
//	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
//	err = vkCreateCommandPool(m_Device, &poolInfo, nullptr, &m_ViewportCommandPool);
//}
//
//
//void RasterView::CreateCommandBuffer()
//{
//	VkResult err;
//
//	VkCommandBufferAllocateInfo allocInfo{};
//	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
//	allocInfo.commandPool = m_ViewportCommandPool;
//	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
//	allocInfo.commandBufferCount = 1;
//	err = vkAllocateCommandBuffers(m_Device, &allocInfo, &m_ViewportCommandBuffer);
//}


void RasterView::RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = m_ViewportRenderPass;
	renderPassInfo.framebuffer = m_ViewportFramebuffers[imageIndex]; /* Might need to change m_ViewportFramebuffers to a single m_ViewportFramebuffer */
	renderPassInfo.renderArea.offset = { 0, 0 };
	renderPassInfo.renderArea.extent = { (uint32_t)m_ViewportSize.x, (uint32_t)m_ViewportSize.y };
	VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;
	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ViewportGraphicsPipeline);

	/* We need to set the viewport and scissor since we said they are dynamic */
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
	scissor.extent = { (uint32_t)m_ViewportSize.x, (uint32_t)m_ViewportSize.y };
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	vkCmdDraw(commandBuffer, 3, 1, 0, 0);

	vkCmdEndRenderPass(commandBuffer);
}