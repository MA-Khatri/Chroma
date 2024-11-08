#include "raster_view.h"

#include <stdlib.h>


RasterView::RasterView()
{
	// TODO?
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
	SetupDebug(app);

	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();
	m_Camera = app->GetMainCamera();
	m_Scenes = app->GetScenes();
	m_Scene = m_Scenes[0];

	InitVulkan();
	m_Scene->VkSetup(m_ViewportSize, m_MSAASampleCount, m_ViewportRenderPass, m_ViewportFramebuffers);
}


void RasterView::OnDetach()
{
	CleanupVulkan();
}


void RasterView::OnUpdate()
{
	/* Update when first switching to new scene */
	int appScene = m_AppHandle->GetSceneID();
	if (appScene != m_SceneID)
	{
		//m_Scene->VkCleanup(); /* Causes a crash? And doesn't seem like we need to. */
		m_SceneID = appScene;
		m_Scene = m_AppHandle->GetScenes()[appScene];
		m_Scene->VkSetup(m_ViewportSize, m_MSAASampleCount, m_ViewportRenderPass, m_ViewportFramebuffers);
	}

	/* We only check for inputs for this view if this view is the currently focused view */
	if (m_AppHandle->m_FocusedWindow == Application::RasterizedViewport)
	{
		/* On hover, check for keyboard/mouse inputs */
		if (m_ViewportHovered)
		{
			/* Set scroll callback for current camera */
			glfwSetWindowUserPointer(m_WindowHandle, m_Camera);
			glfwSetScrollCallback(m_WindowHandle, Camera::ScrollCallback);

			bool updated = m_Camera->Inputs(m_WindowHandle);
		}
		else
		{
			glfwSetScrollCallback(m_WindowHandle, ImGui_ImplGlfw_ScrollCallback);
		}
	}

	/* If UI caused camera params to change... */
	if (m_Camera->m_CameraUIUpdate)
	{
		if (m_Camera->m_ControlMode == CONTROL_MODE_ORBIT)
		{
			m_Camera->UpdateOrbit();
		}

		m_Camera->UpdateViewMatrix();
		m_Camera->UpdateProjectionMatrix();
	}

	/* When switching between viewports... */
	if (m_ViewportFocused && m_AppHandle->m_FocusedWindow != Application::RasterizedViewport)
	{
		m_AppHandle->m_FocusedWindow = Application::RasterizedViewport;
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

/* 
 * Screenshot function for Vulkan based partially on:
 * https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot/screenshot.cpp
 */
std::string RasterView::TakeScreenshot()
{
	uint32_t width = static_cast<int>(m_ViewportSize.x);
	uint32_t height = static_cast<int>(m_ViewportSize.y);

	/* Create a temporary (capture) image to store screenshot data */
	VkImage cptImage;
	VkDeviceMemory cptImageMemory;
	vk::CreateImage(
		width, height,
		1,
		VK_SAMPLE_COUNT_1_BIT,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_TILING_LINEAR,
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		cptImage, cptImageMemory
	);

	/* Get the current viewport image */
	VkImage& srcImage = m_ViewportImages[vk::MainWindowData.FrameIndex];

	/* Transition viewport image to transfer src optimal */
	vk::TransitionImageLayout(
		srcImage,
		vk::MainWindowData.SurfaceFormat.format,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		1
	);

	/* Copy viewport image to cpt image */
	vk::CopyImageToImage(m_ViewportSize, srcImage, cptImage);

	/* Transition viewport image back to color attachment optimal */
	vk::TransitionImageLayout(
		srcImage,
		vk::MainWindowData.SurfaceFormat.format,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		1
	);

	/* Transition cpt image to transfer src optimal */
	vk::TransitionImageLayout(
		cptImage,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		1
	);

	/* Get layout of the image (including row pitch) */
	VkImageSubresource subResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
	VkSubresourceLayout subResourceLayout;
	vkGetImageSubresourceLayout(vk::Device, cptImage, &subResource, &subResourceLayout);

	/* Copy cpt image to host */
	const char* data;
	vkMapMemory(vk::Device, cptImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
	data += subResourceLayout.offset;

	/* Determine if we need to swizzle */
	std::vector<VkFormat> formatsBGR = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SNORM };
	bool swizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), vk::MainWindowData.SurfaceFormat.format) != formatsBGR.end());

	/* Save image to vector with proper format */
	std::vector<uint32_t> pixels;
	pixels.resize(width * height);
	for (uint32_t y = 0; y < height; y++)
	{
		uint32_t* row = (uint32_t*)data;
		for (uint32_t x = 0; x < width; x++)
		{
			uint32_t pixelID = x + y * width;
			uint32_t pixel = 0;
			if (swizzle)
			{
				uint8_t b0, b1, b2, b3;
				uint32_t color = *row;
				b0 = 0xff; /* alpha */
				b1 = (color >> 0 ) & 0xff;
				b2 = (color >> 8 ) & 0xff;
				b3 = (color >> 16) & 0xff;

				pixel = b0 << 24 | b1 << 16 | b2 << 8 | b3 << 0;
			}
			else
			{
				pixel = *row | 0xff000000;
			}
			pixels[pixelID] = pixel;
			row++;
		}
		data += subResourceLayout.rowPitch;
	}

	std::vector<uint32_t> out = RotateAndFlip(pixels, width, height);

	/* Save cpt image to file */
	std::string msg = WriteImageToFile(
		"output/" + GetDateTimeStr() + "_raster.png",
		width, height, 4,
		(void*)out.data(),
		width * 4
	);

	/* Cleanup cpt image */
	vkUnmapMemory(vk::Device, cptImageMemory);
	vkFreeMemory(vk::Device, cptImageMemory, nullptr);
	vkDestroyImage(vk::Device, cptImage, nullptr);

	return msg;
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
	m_Camera->m_ViewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
	m_Camera->m_ViewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
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