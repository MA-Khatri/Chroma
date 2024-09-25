#include "raster_view.h"

#include <stdlib.h>

/* ============================== */
/* === Standard layer methods === */
/* ============================== */

void RasterView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();

	m_Camera = new Camera(100, 100, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 1.0), 45.0f);

	InitVulkan();
	SceneSetup();
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
				ImVec2 newSize = ImGui::GetContentRegionAvail();
				if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y)
				{
					OnResize(newSize);
				}

				VkCommandBuffer commandBuffer = m_AppHandle->GetCommandBuffer();
				RecordCommandBuffer(commandBuffer);
				m_AppHandle->FlushCommandBuffer(commandBuffer);
				m_DescriptorSet = (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(m_Sampler, m_ViewportImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				
				/* Wait until the descriptor set for the viewport image is created */
				vkDeviceWaitIdle(VK::Device);
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


/* =================================== */
/* === RasterView specific methods === */
/* =================================== */

void RasterView::InitVulkan()
{
	/* Set up viewport rendering */
	VK::CreateImage(m_ViewportSize, m_ViewportImage, m_ViewportImageDeviceMemory);
	VK::CreateImageView(m_ViewportImage, m_ViewportImageView);

	VK::CreateRenderPass(m_ViewportRenderPass);
	VK::CreateFrameBuffer(std::vector<VkImageView>{m_ViewportImageView}, m_ViewportRenderPass, m_ViewportSize, m_ViewportFramebuffer);
	VK::CreateSampler(&m_Sampler);
}


void RasterView::CleanupVulkan()
{
	vkDestroyDescriptorPool(VK::Device, m_DescriptorPool, nullptr);
	for (size_t i = 0; i < VK::MinImageCount; i++)
	{
		vkDestroyBuffer(VK::Device, m_UniformBuffers[i], nullptr);
		vkFreeMemory(VK::Device, m_UniformBuffersMemory[i], nullptr);
	}
	vkDestroyDescriptorSetLayout(VK::Device, m_DescriptorSetLayout, nullptr);


	vkDestroySampler(VK::Device, m_Sampler, nullptr);

	vkDestroyFramebuffer(VK::Device, m_ViewportFramebuffer, nullptr);

	auto it = m_Pipelines.begin();
	while (it != m_Pipelines.end())
	{
		vkDestroyPipeline(VK::Device, it->second, nullptr);
	}

	vkDestroyPipelineLayout(VK::Device, m_ViewportPipelineLayout, nullptr);
	vkDestroyRenderPass(VK::Device, m_ViewportRenderPass, nullptr);

	vkDestroyImageView(VK::Device, m_ViewportImageView, nullptr);
	vkDestroyImage(VK::Device, m_ViewportImage, nullptr);
	vkFreeMemory(VK::Device, m_ViewportImageDeviceMemory, nullptr);
}


void RasterView::OnResize(ImVec2 newSize)
{
	m_ViewportSize = newSize;

	vkDeviceWaitIdle(VK::Device);

	vkDestroyFramebuffer(VK::Device, m_ViewportFramebuffer, nullptr);
	vkDestroyImageView(VK::Device, m_ViewportImageView, nullptr);
	vkDestroyImage(VK::Device, m_ViewportImage, nullptr);
	
	VK::CreateImage(m_ViewportSize, m_ViewportImage, m_ViewportImageDeviceMemory);
	VK::CreateImageView(m_ViewportImage, m_ViewportImageView);
	VK::CreateFrameBuffer(std::vector<VkImageView>{m_ViewportImageView}, m_ViewportRenderPass, m_ViewportSize, m_ViewportFramebuffer);
}


void RasterView::SceneSetup()
{
	/* Descriptor sets: uniforms, textures */
	VkDescriptorSetLayoutBinding mvpLayoutBinding{};
	mvpLayoutBinding.binding = 0;
	mvpLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	mvpLayoutBinding.descriptorCount = 1;
	mvpLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	mvpLayoutBinding.pImmutableSamplers = nullptr; /* optional -- for texture samplers */
	VK::CreateDescriptorSetLayout(mvpLayoutBinding, m_DescriptorSetLayout);
	VK::CreateUniformBuffers(sizeof(UniformBufferObject), m_UniformBuffers, m_UniformBuffersMemory, m_UniformBuffersMapped);
	
	VK::CreateDescriptorPool(m_DescriptorPool);
	VK::CreateDescriptorSets(m_DescriptorSetLayout, m_DescriptorPool, m_DescriptorSets);
	for (size_t i = 0; i < VK::MinImageCount; i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = m_UniformBuffers[i];
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformBufferObject);

		VkWriteDescriptorSet descriptorWrite{};
		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = m_DescriptorSets[i];
		descriptorWrite.dstBinding = 0;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo = &bufferInfo;
		descriptorWrite.pImageInfo = nullptr; /* optional */
		descriptorWrite.pTexelBufferView = nullptr; /* optional */
		vkUpdateDescriptorSets(VK::Device, 1, &descriptorWrite, 0, nullptr);
	}

	/* Generate graphics pipelines with different shaders */
	std::vector<std::string> shadersBasic = { "res/shaders/Basic.vert", "res/shaders/Basic.frag" };
	//std::vector<std::string> shadersBasic = { "res/shaders/VkTut.vert", "res/shaders/VkTut.frag" };
	//std::vector<std::string> shadersBasic = { "res/shaders/Basic.vert.spv", "res/shaders/Basic.frag.spv" };
	//std::vector<std::string> shadersBasic = { "res/shaders/VkTut.vert.spv", "res/shaders/VkTut.frag.spv" };
	m_Pipelines[Basic] = VK::CreateGraphicsPipeline(shadersBasic, m_ViewportSize, m_ViewportRenderPass, m_DescriptorSetLayout, m_ViewportPipelineLayout);

	/* Create objects that will be drawn */
	Object* triangle = new Object(CreateHelloTriangle(), m_Pipelines[Basic]);
	m_Objects.push_back(triangle);

	Object* plane = new Object(CreatePlane(), m_Pipelines[Basic]);
	m_Objects.push_back(plane);
}


void RasterView::UpdateUniformBuffer(uint32_t currentImage)
{
	UniformBufferObject ubo{};
	ubo.proj = m_Camera->projection_matrix;
	ubo.view = m_Camera->view_matrix;
	ubo.model = glm::scale(glm::vec3(10.0f, 10.0f, 10.0f));

	/* We need to flip y in the proj mat to convert from OpenGL clip coordinate convention to Vulkan convention */
	ubo.proj[1][1] *= -1;

	memcpy(m_UniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}


void RasterView::RecordCommandBuffer(VkCommandBuffer& commandBuffer)
{
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = m_ViewportRenderPass;
	renderPassInfo.framebuffer = m_ViewportFramebuffer;
	renderPassInfo.renderArea.offset = { 0, 0 };
	renderPassInfo.renderArea.extent = { (uint32_t)m_ViewportSize.x, (uint32_t)m_ViewportSize.y };
	VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;
	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

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

	/* Update uniforms */
	UpdateUniformBuffer(VK::MainWindowData.FrameIndex);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ViewportPipelineLayout, 0, 1, &m_DescriptorSets[VK::MainWindowData.FrameIndex], 0, nullptr);

	/* Draw the objects */
	for (auto object : m_Objects)
	{
		object->Draw(commandBuffer);
	}

	vkCmdEndRenderPass(commandBuffer);
}