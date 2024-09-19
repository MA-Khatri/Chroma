#include "raster_view.h"

static uint32_t GetVulkanMemoryType(VkMemoryPropertyFlags properties, uint32_t type_bits)
{
	VkPhysicalDeviceMemoryProperties prop;
	vkGetPhysicalDeviceMemoryProperties(Application::GetPhysicalDevice(), &prop);
	for (uint32_t i = 0; i < prop.memoryTypeCount; i++)
	{
		if ((prop.memoryTypes[i].propertyFlags & properties) == properties && type_bits & (1 << i))
		{
			return i;
		}
	}

	return 0xffffffff;
}


void RasterView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();

	m_PhysicalDevice = app->GetPhysicalDevice();
	m_Device = app->GetDevice();
	m_MinImageCount = app->GetMinImageCount();

	CreateViewportImages();
	CreateViewportImageViews();

	m_Camera = new Camera(100, 100, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 1.0), 45.0f);

	m_Image = std::make_shared<Image>("./res/textures/teapot_normal.png");
}


void RasterView::OnDetach()
{
	for (auto imageView : m_ViewportImageViews)
	{
		vkDestroyImageView(m_Device, imageView, nullptr);
	}

	for (auto image : m_ViewportImages)
	{
		vkDestroyImage(m_Device, image, nullptr);
	}
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
				ImVec2 tempSize = ImGui::GetWindowSize();
				if (m_ViewportSize.x != tempSize.x || m_ViewportSize.y != tempSize.y)
				{
					OnResize(tempSize);
				}

				ImGui::Image(m_Image->GetDescriptorSet(), { (float)m_Image->GetWidth(), (float)m_Image->GetHeight() });
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


void RasterView::OnResize(ImVec2 newSize)
{
	m_ViewportSize = newSize;

	/* TODO: Vulkan stuff... */
}


void RasterView::CreateViewportImages()
{
	VkResult err;

	m_ViewportImages.resize(m_MinImageCount);
	m_ImageDeviceMemory.resize(m_MinImageCount);

	for (uint32_t i = 0; i < m_MinImageCount; i++)
	{
		VkImageCreateInfo imageCreateInfo = {};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
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
		check_vk_result(err);

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(m_Device, m_ViewportImages[i], &memRequirements);

		VkMemoryAllocateInfo memAllocInfo = {};
		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllocInfo.allocationSize = memRequirements.size;
		memAllocInfo.memoryTypeIndex = GetVulkanMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements.memoryTypeBits);
		err = vkAllocateMemory(m_Device, &memAllocInfo, nullptr, &m_Memory);
		check_vk_result(err);

		err = vkBindImageMemory(m_Device, m_ViewportImages[i], m_Memory, 0);
		check_vk_result(err);
	}
}


void RasterView::CreateViewportImageViews()
{
	VkResult err;

	m_ViewportImageViews.resize(m_MinImageCount);

	for (uint32_t i = 0; i < m_MinImageCount; i++)
	{
		VkImageViewCreateInfo imageViewCreateInfo = {};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.image = m_ViewportImages[i];
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.layerCount = 1;
		err = vkCreateImageView(m_Device, &imageViewCreateInfo, nullptr, &m_ViewportImageViews[i]);
		check_vk_result(err);
	}
}
