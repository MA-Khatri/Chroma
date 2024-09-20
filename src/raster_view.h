#pragma once

#include <vector>
#include <optional>
#include "vulkan/vulkan.h"

#include "layer.h"
#include "image.h"
#include "camera.h"

class RasterView : public Layer 
{
public:

	/* Standard layer methods */
	virtual void OnAttach(Application* app);
	virtual void OnDetach();
	virtual void OnUpdate();
	virtual void OnUIRender();

	/* RasterView specific methods -- many of these should probably be moved and improved... */
	void InitVulkan();
	void CleanupVulkan();
	void OnResize(ImVec2 newSize);
	void CreateViewportImages();
	void CreateViewportImageViews();
	void CreateRenderPass();
	void CreateGraphicsPipeline();
	void CreateFrameBuffers();
	void CreateSampler();
	//void CreateCommandPool();
	//void CreateCommandBuffer();
	void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

	VkShaderModule CreateShaderModule(const std::vector<char>& code);

	/* This struct & function and many other functions in this class are probably better established somewhere else... */
	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete()
		{
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};
	QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera;
	bool m_ViewportFocused = false;
	ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

	uint32_t m_MinImageCount;
	VkPhysicalDevice m_PhysicalDevice;
	VkDevice m_Device;
	std::vector<VkImage> m_ViewportImages;
	std::vector<VkDeviceMemory> m_ImageDeviceMemory;
	std::vector<VkImageView> m_ViewportImageViews;
	VkRenderPass m_ViewportRenderPass;
	VkPipelineLayout m_ViewportPipelineLayout;
	VkPipeline m_ViewportGraphicsPipeline;
	std::vector<VkFramebuffer> m_ViewportFramebuffers;
	//VkCommandPool m_ViewportCommandPool;
	//std::vector<VkCommandBuffer> m_ViewportCommandBuffers;
	//VkCommandBuffer m_ViewportCommandBuffer;
	VkSampler m_Sampler;
	VkDescriptorSet m_DescriptorSet;

	std::shared_ptr<Image> m_Image;
};
