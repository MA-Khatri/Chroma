#include "raster_view.h"

#include <stdlib.h>


RasterView::RasterView(std::shared_ptr<Scene> scene)
{
	m_Scene = scene;
}

RasterView::~RasterView()
{
	// TODO?
}


/* ============================== */
/* === Standard layer methods === */
/* ============================== */

void RasterView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();
	m_Camera = app->GetMainCamera();

	InitVulkan();
	m_Scene->VkSetup(m_ViewportSize, m_MSAASampleCount, m_ViewportRenderPass, m_ViewportFramebuffers);
}


void RasterView::OnDetach()
{
	CleanupVulkan();
}


void RasterView::OnUpdate()
{
	if (m_ViewportFocused) m_AppHandle->m_FocusedWindow = Application::RasterizedViewport;

	if (m_ViewportHovered)
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
			m_ViewportFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

			ImGui::BeginChild("Rasterized");
			{
				m_ViewportHovered = ImGui::IsWindowHovered();

				ImVec2 newSize = ImGui::GetContentRegionAvail();
				if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y)
				{
					OnResize(newSize);
				}

				if (m_AppHandle->m_FocusedWindow == Application::RasterizedViewport)
				{
					m_Scene->VkDraw(*m_Camera);

					/* Wait until the descriptor set for the viewport image is created */
					/* This could be a source of latency later on -- might be better to add multiple images here as well to allow simultaneous rendering/displaying */
					vkDeviceWaitIdle(vk::Device);

					/* Note: we flip the image vertically to match Vulkan convention! */
					ImGui::Image(m_ViewportImageDescriptorSets[vk::MainWindowData.FrameIndex], m_ViewportSize, ImVec2(0, 1), ImVec2(1, 0));
				}
			}
			ImGui::EndChild();
		}
		ImGui::End();
	}
	/* Add back in padding for non-viewport ImGui */
	ImGui::PopStyleVar();

	ImGui::ShowDemoWindow();

	if (m_AppHandle->m_FocusedWindow == Application::RasterizedViewport)
	{
		ImGui::Begin("Debug Panel");
		{
			CommonDebug(m_AppHandle, m_ViewportSize, *m_Camera);
		}
		ImGui::End();
	}
}


void RasterView::TakeScreenshot()
{
	/* Create a temporary image to store screenshot data */
	VkImage cptImage;
	VkDeviceMemory cptImageMemory;
	vk::CreateImage(
		static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), 
		1,
		VK_SAMPLE_COUNT_1_BIT, 
		VK_FORMAT_R8G8B8A8_UNORM, 
		VK_IMAGE_TILING_LINEAR, 
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
		cptImage, cptImageMemory
	);

	VkImage& srcImage = m_ViewportImages[vk::MainWindowData.FrameIndex];

	VkCommandBuffer commandBuffer = vk::GetGraphicsCommandBuffer();
	{
		/* Transition viewport image to transfer src optimal */
		vk::TransitionImageLayout(
			commandBuffer,
			srcImage, 
			vk::MainWindowData.SurfaceFormat.format,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
			1
		);

		/* Copy viewport image to cpt image */
		vk::CopyImageToImage(commandBuffer, m_ViewportSize, srcImage, cptImage);

		/* Transition viewport image back to color attachment optimal */
		vk::TransitionImageLayout(
			commandBuffer,
			srcImage,
			vk::MainWindowData.SurfaceFormat.format,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			1
		);

		/* Transition cpt image to transfer src optimal */
		vk::TransitionImageLayout(
			commandBuffer,
			cptImage,
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			1
		);
	}
	vk::FlushGraphicsCommandBuffer(commandBuffer);

	/* Copy cpt image to host */
	uint32_t size = static_cast<uint32_t>(m_ViewportSize.x) * static_cast<uint32_t>(m_ViewportSize.y) * 4;
	void* data;
	vkMapMemory(vk::Device, cptImageMemory, 0, size, 0, &data);

	/* Save cpt image to file */
	WriteImageToFile(
		data, 
		static_cast<int>(m_ViewportSize.x), static_cast<int>(m_ViewportSize.y), 
		"output/" + GetDateTimeStr() + "_raster.png"
	);

	/* Cleanup cpt image */
	vkUnmapMemory(vk::Device, cptImageMemory);
	vkFreeMemory(vk::Device, cptImageMemory, nullptr);
	vkDestroyImage(vk::Device, cptImage, nullptr);
}


/* =================================== */
/* === RasterView specific methods === */
/* =================================== */

void RasterView::InitVulkan()
{
	m_MSAASampleCount = vk::MaxMSAASamples;

	/* Set up viewport rendering */
	vk::CreateRenderPass(m_MSAASampleCount, m_ViewportRenderPass);
	vk::CreateViewportSampler(&m_ViewportSampler);

	vk::CreateColorResources(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount, m_ColorImage, m_ColorImageMemory, m_ColorImageView);
	vk::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount, m_DepthImage, m_DepthImageMemory, m_DepthImageView);
	CreateViewportImagesAndFramebuffers();
	CreateViewportImageDescriptorSets();
}


void RasterView::CleanupVulkan()
{
	vkDestroySampler(vk::Device, m_ViewportSampler, nullptr);

	DestroyColorResources();
	DestroyDepthResources();

	DestroyViewportImageDescriptorSets();
	DestroyViewportImagesAndFramebuffers();

	vkDestroyRenderPass(vk::Device, m_ViewportRenderPass, nullptr);
}


void RasterView::OnResize(ImVec2 newSize)
{
	m_ViewportSize = newSize;

	ImVec2 mainWindowPos = ImGui::GetMainViewport()->Pos;
	ImVec2 viewportPos = ImGui::GetWindowPos();
	ImVec2 rPos = ImVec2(viewportPos.x - mainWindowPos.x, viewportPos.y - mainWindowPos.y);
	ImVec2 minR = ImGui::GetWindowContentRegionMin();
	ImVec2 maxR = ImGui::GetWindowContentRegionMax();
	m_Camera->viewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
	m_Camera->viewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
	m_Camera->UpdateProjectionMatrix(static_cast<int>(m_ViewportSize.x), static_cast<int>(m_ViewportSize.y));

	/* Before re-creating the images, we MUST wait for the device to be done using them */
	vkDeviceWaitIdle(vk::Device);

	/* Cleanup previous */
	DestroyColorResources();
	DestroyDepthResources();
	DestroyViewportImagesAndFramebuffers();
	DestroyViewportImageDescriptorSets();

	/* Recreate new */
	vk::CreateColorResources(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount, m_ColorImage, m_ColorImageMemory, m_ColorImageView);
	vk::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount, m_DepthImage, m_DepthImageMemory, m_DepthImageView);
	CreateViewportImagesAndFramebuffers();
	CreateViewportImageDescriptorSets();

	m_Scene->VkResize(m_ViewportSize, m_ViewportFramebuffers);
}


void RasterView::CreateViewportImagesAndFramebuffers()
{
	vk::CreateViewportImages(vk::ImageCount, m_ViewportSize, m_ViewportImages, m_ViewportImagesDeviceMemory);
	vk::CreateViewportImageViews(m_ViewportImages, m_ViewportImageViews);
	m_ViewportFramebuffers.resize(vk::ImageCount);
	for (uint32_t i = 0; i < vk::ImageCount; i++)
	{
		vk::CreateFrameBuffer(std::vector<VkImageView>{m_ColorImageView, m_DepthImageView, m_ViewportImageViews[i]}, m_ViewportRenderPass, m_ViewportSize, m_ViewportFramebuffers[i]);
	}
}


void RasterView::DestroyViewportImagesAndFramebuffers()
{
	for (uint32_t i = 0; i < vk::ImageCount; i++)
	{
		vkDestroyFramebuffer(vk::Device, m_ViewportFramebuffers[i], nullptr);
	}

	for (uint32_t i = 0; i < vk::ImageCount; i++)
	{
		vkDestroyImageView(vk::Device, m_ViewportImageViews[i], nullptr);
		vkDestroyImage(vk::Device, m_ViewportImages[i], nullptr);
		vkFreeMemory(vk::Device, m_ViewportImagesDeviceMemory[i], nullptr);
	}
	/* 
	 * Need to clear the memory vector otherwise we may get errors saying that 
	 * we are trying to free already freed memory if we call CreateImage() after.
	 */
	m_ViewportImagesDeviceMemory.clear();
}


void RasterView::CreateViewportImageDescriptorSets()
{
	for (uint32_t i = 0; i < vk::ImageCount; i++)
	{
		m_ViewportImageDescriptorSets.push_back((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(m_ViewportSampler, m_ViewportImageViews[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	}
}


void RasterView::DestroyViewportImageDescriptorSets()
{
	for (auto& descriptorSet : m_ViewportImageDescriptorSets)
	{
		ImGui_ImplVulkan_RemoveTexture(descriptorSet);
	}

	m_ViewportImageDescriptorSets.clear();
}


void RasterView::DestroyColorResources()
{
	vkDestroyImageView(vk::Device, m_ColorImageView, nullptr);
	vkDestroyImage(vk::Device, m_ColorImage, nullptr);
	vkFreeMemory(vk::Device, m_ColorImageMemory, nullptr);
}


void RasterView::DestroyDepthResources()
{
	vkDestroyImageView(vk::Device, m_DepthImageView, nullptr);
	vkDestroyImage(vk::Device, m_DepthImage, nullptr);
	vkFreeMemory(vk::Device, m_DepthImageMemory, nullptr);
}