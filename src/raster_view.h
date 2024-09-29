#pragma once

#include <vector>
#include <map>
#include <optional>
#include "vulkan/vulkan.h"

#include "vulkan_utils.h"
#include "layer.h"
#include "image.h"
#include "camera.h"
#include "mesh.h"
#include "object.h"


class RasterView : public Layer 
{
public:

	/* Standard layer methods */
	virtual void OnAttach(Application* app);
	virtual void OnDetach();
	virtual void OnUpdate();
	virtual void OnUIRender();


private:

	/* RasterView specific methods */
	void InitVulkan();
	void CleanupVulkan();
	void OnResize(ImVec2 newSize);

	void SceneSetup();
	void CreateViewportImageDescriptorSets();
	void DestroyViewportImageDescriptorSets();
	void CreateViewportImagesAndFramebuffers();
	void DestroyViewportImagesAndFramebuffers();

	void DestroyColorResources();
	void DestroyDepthResources();

	void RecordCommandBuffer(VkCommandBuffer& commandBuffer);


private:

	/* List of pipeline types */
	enum Pipelines
	{
		Basic,
	};

	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera;
	bool m_ViewportFocused = false;
	bool m_ViewportHovered = false;
	ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

	std::vector<VkImage> m_ViewportImages;
	std::vector<VkDeviceMemory> m_ViewportImagesDeviceMemory;
	std::vector<VkImageView> m_ViewportImageViews;
	std::vector<VkDescriptorSet> m_ViewportImageDescriptorSets;

	VkSampleCountFlagBits m_MSAASampleCount = VK_SAMPLE_COUNT_1_BIT;
	VkImage m_ColorImage; /* for MSAA */
	VkDeviceMemory m_ColorImageMemory;
	VkImageView m_ColorImageView;

	VkRenderPass m_ViewportRenderPass;
	VkPipelineLayout m_ViewportPipelineLayout;
	std::vector<VkFramebuffer> m_ViewportFramebuffers;
	VkSampler m_ViewportSampler;

	VkImage m_DepthImage;
	VkDeviceMemory m_DepthImageMemory;
	VkImageView m_DepthImageView;
	
	std::map<Pipelines, PipelineInfo> m_Pipelines; /* Pipelines with diff. shaders/draw modes */
	
	std::vector<Object*> m_Objects; /* Objects to be drawn */

	/* We set the camera's view and projection matrices as push constants */
	struct PushConstants {
		alignas(16) glm::mat4 view = glm::mat4(1.0f);
		alignas(16) glm::mat4 proj = glm::mat4(1.0f);
	};

	VkDescriptorSetLayout m_DescriptorSetLayout;
	VkDescriptorPool m_DescriptorPool;
};
