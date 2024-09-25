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

	/* RasterView specific methods */
	void InitVulkan();
	void CleanupVulkan();
	void OnResize(ImVec2 newSize);

	void SceneSetup();

	void UpdateUniformBuffer(uint32_t currentImage);

	void RecordCommandBuffer(VkCommandBuffer& commandBuffer);

	/* List of pipeline types */
	enum Pipelines
	{
		Basic,
	};

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera;
	bool m_ViewportFocused = false;
	bool m_ViewportHovered = false;
	ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

	VkImage m_ViewportImage;
	VkDeviceMemory m_ViewportImageDeviceMemory;
	VkImageView m_ViewportImageView;

	VkRenderPass m_ViewportRenderPass;
	VkPipelineLayout m_ViewportPipelineLayout;
	VkFramebuffer m_ViewportFramebuffer;
	VkSampler m_Sampler;
	VkDescriptorSet m_DescriptorSet;

	std::map<Pipelines, VkPipeline> m_Pipelines; /* Pipelines with diff. shaders/draw modes */
	
	std::vector<Object*> m_Objects; /* Objects to be drawn */



	struct UniformBufferObject {
		alignas(16) glm::mat4 model = glm::mat4(1.0f);
		alignas(16) glm::mat4 view = glm::mat4(1.0f);
		alignas(16) glm::mat4 proj = glm::mat4(1.0f);
	};

	VkDescriptorSetLayout m_DescriptorSetLayout;

	std::vector<VkBuffer> m_UniformBuffers;
	std::vector<VkDeviceMemory> m_UniformBuffersMemory;
	std::vector<void*> m_UniformBuffersMapped;
	VkDescriptorPool m_DescriptorPool;
	std::vector<VkDescriptorSet> m_DescriptorSets;
};
