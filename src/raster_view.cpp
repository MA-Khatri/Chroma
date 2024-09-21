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
				/* The IsWindowHovered check is to prevent runaway memory leaks from the OnResize function. I.e., we limit the calls to OnResize. */
				if (ImGui::IsWindowHovered() && (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y))
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
	VK::CreateImage(m_ViewportSize, &m_ViewportImage, &m_ImageDeviceMemory);
	VK::CreateImageView(&m_ViewportImage, &m_ViewportImageView);
	VK::CreateRenderPass(&m_ViewportRenderPass);
	VK::CreateGraphicsPipeline("res/shaders/spv/HelloTriangle.vert.spv", "res/shaders/spv/HelloTriangle.frag.spv", m_ViewportSize, &m_ViewportRenderPass, &m_ViewportPipelineLayout, &m_ViewportGraphicsPipeline);
	VK::CreateFrameBuffer(std::vector<VkImageView>{m_ViewportImageView}, & m_ViewportRenderPass, m_ViewportSize, &m_ViewportFramebuffer);
	VK::CreateSampler(&m_Sampler);
}


void RasterView::CleanupVulkan()
{
	vkDestroySampler(VK::Device, m_Sampler, nullptr);

	vkDestroyFramebuffer(VK::Device, m_ViewportFramebuffer, nullptr);

	vkDestroyPipeline(VK::Device, m_ViewportGraphicsPipeline, nullptr);
	vkDestroyPipelineLayout(VK::Device, m_ViewportPipelineLayout, nullptr);
	vkDestroyRenderPass(VK::Device, m_ViewportRenderPass, nullptr);

	vkDestroyImageView(VK::Device, m_ViewportImageView, nullptr);

	vkDestroyImage(VK::Device, m_ViewportImage, nullptr);
}


void RasterView::OnResize(ImVec2 newSize)
{
	m_ViewportSize = newSize;

	vkDeviceWaitIdle(VK::Device);

	vkDestroyFramebuffer(VK::Device, m_ViewportFramebuffer, nullptr);
	vkDestroyImageView(VK::Device, m_ViewportImageView, nullptr);
	vkDestroyImage(VK::Device, m_ViewportImage, nullptr);
	
	VK::CreateImage(m_ViewportSize, &m_ViewportImage, &m_ImageDeviceMemory);
	VK::CreateImageView(&m_ViewportImage, &m_ViewportImageView);
	VK::CreateFrameBuffer(std::vector<VkImageView>{m_ViewportImageView}, &m_ViewportRenderPass, m_ViewportSize, &m_ViewportFramebuffer);
}


void RasterView::RecordCommandBuffer(VkCommandBuffer commandBuffer)
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