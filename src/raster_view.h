#pragma once

#include <vector>
#include "vulkan/vulkan.h"

#include "layer.h"
#include "image.h"
#include "camera.h"

class RasterView : public Layer {

	/* Standard layer methods */
	virtual void OnAttach(Application* app);
	virtual void OnDetach();
	virtual void OnUpdate();
	virtual void OnUIRender();

	/* RasterView specific methods */
	void OnResize(ImVec2 newSize);
	void CreateViewportImages();
	void CreateViewportImageViews();
	void CreateGraphicsPipeline();
	VkShaderModule CreateShaderModule(const std::vector<char>& code);

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera;
	bool m_ViewportFocused = false;
	ImVec2 m_ViewportSize = ImVec2(10.0f, 10.0f);

	uint32_t m_MinImageCount;
	VkPhysicalDevice m_PhysicalDevice;
	VkDevice m_Device;
	std::vector<VkImage> m_ViewportImages;
	std::vector<VkDeviceMemory> m_ImageDeviceMemory;
	std::vector<VkImageView> m_ViewportImageViews;
	VkRenderPass m_ViewportRenderPass;
	VkPipeline m_ViewportPipeline;
	VkCommandPool m_ViewportCommandPool;
	std::vector<VkFramebuffer> m_ViewportFramebuffers;
	std::vector<VkCommandBuffer> m_ViewportCommandBuffers;

	std::shared_ptr<Image> m_Image;
};
