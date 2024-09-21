#pragma once

#include <vector>
#include <optional>
#include "vulkan/vulkan.h"

#include "vulkan_utils.h"
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

	void RecordCommandBuffer(VkCommandBuffer commandBuffer);

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera;
	bool m_ViewportFocused = false;
	ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

	VkImage m_ViewportImage;
	VkDeviceMemory m_ViewportImageDeviceMemory;
	VkImageView m_ViewportImageView;

	VkRenderPass m_ViewportRenderPass;
	VkPipelineLayout m_ViewportPipelineLayout;
	VkPipeline m_ViewportGraphicsPipeline;
	VkFramebuffer m_ViewportFramebuffer;
	VkSampler m_Sampler;
	VkDescriptorSet m_DescriptorSet;

	/* later should be multiple buffers for each mesh? */
	std::vector<Vertex> m_Vertices;
	VkBuffer m_VertexBuffer;
	VkDeviceMemory m_VertexBufferMemory;
};
