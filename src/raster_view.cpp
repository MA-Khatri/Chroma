#include "raster_view.h"

#include <stdlib.h>

/* ============================== */
/* === Standard layer methods === */
/* ============================== */

void RasterView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();

	m_Camera = new Camera(100, 100, glm::vec3(0.0f, 10.0f, 5.0f), glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, 1.0), 45.0f);

	InitVulkan();
	SceneSetup();
}


void RasterView::OnDetach()
{
	CleanupVulkan();
}


void RasterView::OnUpdate()
{
	if (m_ViewportHovered)
	{
		m_Camera->Inputs(m_WindowHandle);
	}

	//UpdateUniformBuffer(VK::MainWindowData.FrameIndex);
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
				m_ViewportHovered = ImGui::IsWindowHovered();

				ImVec2 newSize = ImGui::GetContentRegionAvail();
				if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y)
				{
					OnResize(newSize);
				}

				VkCommandBuffer commandBuffer = VK::GetGraphicsCommandBuffer();
				RecordCommandBuffer(commandBuffer);
				VK::FlushGraphicsCommandBuffer(commandBuffer);

				/* Wait until the descriptor set for the viewport image is created */
				/* This could be a source of latency later on -- might be better to add multiple images here as well to allow simultaneous rendering/displaying */
				vkDeviceWaitIdle(VK::Device);
				ImGui::Image(m_ViewportImageDescriptorSets[VK::MainWindowData.FrameIndex], m_ViewportSize);
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


/* =================================== */
/* === RasterView specific methods === */
/* =================================== */

void RasterView::InitVulkan()
{
	/* Set up viewport rendering */
	VK::CreateRenderPass(m_ViewportRenderPass);
	VK::CreateViewportSampler(&m_ViewportSampler);

	VK::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), m_DepthImage, m_DepthImageMemory, m_DepthImageView);
	CreateImagesAndFramebuffers();
	CreateViewportImageDescriptorSets();
}


void RasterView::CleanupVulkan()
{
	vkDestroyImageView(VK::Device, m_DepthImageView, nullptr);
	vkDestroyImage(VK::Device, m_DepthImage, nullptr);
	vkFreeMemory(VK::Device, m_DepthImageMemory, nullptr);

	//vkDestroySampler(VK::Device, m_TextureSampler, nullptr);
	//vkDestroyImageView(VK::Device, m_TextureImageView, nullptr);
	//vkDestroyImage(VK::Device, m_TextureImage, nullptr);
	//vkFreeMemory(VK::Device, m_TextureImageMemory, nullptr);

	vkDestroyDescriptorPool(VK::Device, m_DescriptorPool, nullptr);
	//for (size_t i = 0; i < VK::ImageCount; i++)
	//{
	//	vkDestroyBuffer(VK::Device, m_UniformBuffers[i], nullptr);
	//	vkFreeMemory(VK::Device, m_UniformBuffersMemory[i], nullptr);
	//}
	vkDestroyDescriptorSetLayout(VK::Device, m_DescriptorSetLayout, nullptr);

	vkDestroySampler(VK::Device, m_ViewportSampler, nullptr);

	DestroyImagesAndFramebuffers();

	auto it = m_Pipelines.begin();
	while (it != m_Pipelines.end())
	{
		vkDestroyPipeline(VK::Device, it->second, nullptr);
	}

	vkDestroyPipelineLayout(VK::Device, m_ViewportPipelineLayout, nullptr);
	vkDestroyRenderPass(VK::Device, m_ViewportRenderPass, nullptr);
}


void RasterView::OnResize(ImVec2 newSize)
{
	ImVec2 mainWindowPos = ImGui::GetMainViewport()->Pos;
	ImVec2 viewportPos = ImGui::GetWindowPos();
	ImVec2 rPos = ImVec2(viewportPos.x - mainWindowPos.x, viewportPos.y - mainWindowPos.y);
	ImVec2 minR = ImGui::GetWindowContentRegionMin();
	ImVec2 maxR = ImGui::GetWindowContentRegionMax();
	m_Camera->viewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
	m_Camera->viewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
	//std::cout << m_Camera->viewportContentMin.x << " " << m_Camera->viewportContentMin.y << "   " << m_Camera->viewportContentMax.x << " " << m_Camera->viewportContentMax.y << std::endl;
	m_ViewportSize = newSize;
	m_Camera->UpdateProjectionMatrix((int)m_ViewportSize.x, (int)m_ViewportSize.y);

	/* Before re-creating the images, we MUST wait for the device to be done using them */
	vkDeviceWaitIdle(VK::Device);
	VK::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), m_DepthImage, m_DepthImageMemory, m_DepthImageView);
	CreateImagesAndFramebuffers();
	CreateViewportImageDescriptorSets();
}


void RasterView::SceneSetup()
{
	/* Descriptor set layout creation: uniforms, textures/samplers */
	std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.pImmutableSamplers = nullptr;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	layoutBindings.push_back(uboLayoutBinding);
	
	VkDescriptorSetLayoutBinding samplerLayoutBinding{};
	samplerLayoutBinding.binding = 1;
	samplerLayoutBinding.descriptorCount = 1;
	samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	samplerLayoutBinding.pImmutableSamplers = nullptr;
	samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	layoutBindings.push_back(samplerLayoutBinding);

	VK::CreateDescriptorSetLayout(layoutBindings, m_DescriptorSetLayout);
	VK::CreateDescriptorPool(2, m_DescriptorPool); /* Make sure to update the max number of descriptor sets! */

	/* Generate graphics pipelines with different shaders */
	std::vector<std::string> shadersBasic = { "res/shaders/Basic.vert", "res/shaders/Basic.frag" };
	//std::vector<std::string> shadersBasic = { "res/shaders/Basic.vert.spv", "res/shaders/Basic.frag.spv" };
	m_Pipelines[Basic] = VK::CreateGraphicsPipeline(shadersBasic, m_ViewportSize, m_ViewportRenderPass, m_DescriptorSetLayout, m_ViewportPipelineLayout);

	/* Create objects that will be drawn */
	Object* triangle = new Object(CreateHelloTriangle(), m_DescriptorSetLayout, m_DescriptorPool, m_ViewportPipelineLayout, m_Pipelines[Basic]);
	m_Objects.push_back(triangle);

	Object* plane = new Object(CreatePlane(), m_DescriptorSetLayout, m_DescriptorPool, m_ViewportPipelineLayout, m_Pipelines[Basic]);
	m_Objects.push_back(plane);
}


void RasterView::CreateImagesAndFramebuffers()
{
	VK::CreateImages(VK::ImageCount, m_ViewportSize, m_ViewportImages, m_ViewportImagesDeviceMemory);
	VK::CreateImageViews(m_ViewportImages, m_ViewportImageViews);
	m_ViewportFramebuffers.resize(VK::ImageCount);
	for (uint32_t i = 0; i < VK::ImageCount; i++)
	{
		VK::CreateFrameBuffer(std::vector<VkImageView>{m_ViewportImageViews[i], m_DepthImageView}, m_ViewportRenderPass, m_ViewportSize, m_ViewportFramebuffers[i]);
	}
}


void RasterView::DestroyImagesAndFramebuffers()
{
	for (uint32_t i = 0; i < VK::ImageCount; i++)
	{
		vkDestroyFramebuffer(VK::Device, m_ViewportFramebuffers[i], nullptr);
	}

	for (uint32_t i = 0; i < VK::ImageCount; i++)
	{
		vkDestroyImageView(VK::Device, m_ViewportImageViews[i], nullptr);
		vkDestroyImage(VK::Device, m_ViewportImages[i], nullptr);
		vkFreeMemory(VK::Device, m_ViewportImagesDeviceMemory[i], nullptr);
	}
	/* 
	 * Need to clear the memory vector otherwise we may get errors saying that 
	 * we are trying to free already freed memory if we call CreateImage() after.
	 */
	m_ViewportImagesDeviceMemory.clear();
}


void RasterView::CreateViewportImageDescriptorSets()
{
	m_ViewportImageDescriptorSets.clear();

	for (uint32_t i = 0; i < VK::ImageCount; i++)
	{
		m_ViewportImageDescriptorSets.push_back((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(m_ViewportSampler, m_ViewportImageViews[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	}
}


void RasterView::RecordCommandBuffer(VkCommandBuffer& commandBuffer)
{
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = m_ViewportRenderPass;
	renderPassInfo.framebuffer = m_ViewportFramebuffers[VK::MainWindowData.FrameIndex];
	renderPassInfo.renderArea.offset = { 0, 0 };
	renderPassInfo.renderArea.extent = { static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y) };

	std::array<VkClearValue, 2> clearValues{};
	clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
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
	constants.view = m_Camera->view_matrix;
	constants.proj = m_Camera->projection_matrix;
	constants.proj[1][1] *= -1; /* Flip y in proj matrix to match Vulkan convention */
	vkCmdPushConstants(commandBuffer, m_ViewportPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &constants);

	/* Draw the objects */
	for (auto object : m_Objects)
	{
		object->Draw(commandBuffer);
	}

	vkCmdEndRenderPass(commandBuffer);
}