#pragma once

#include <map>

#include "vulkan_utils.h"
#include "object.h"
#include "mesh.h"
#include "camera.h"
#include "application.h"


/* Forward declarations */
class Object;
class PipelineInfo;


class Scene
{
public:
	Scene();
	~Scene();

	/* Utility functions for scene setup and rendering with Vulkan */
	void VkSetup(ImVec2 viewportSize, VkSampleCountFlagBits sampleCount, VkRenderPass& renderPass, std::vector<VkFramebuffer>& framebuffers);
	void VkResize(ImVec2 newSize, std::vector<VkFramebuffer>& framebuffers);
	void VkDraw(const Camera& camera);
	void VkCleanup();

public:
	/* Scene clear/background color */
	glm::vec3 m_ClearColor = glm::vec3(63.0f / 255.0f, 63.0f / 255.0f, 63.0f / 255.0f);

	/* List of pipeline types */
	static enum PipelineType
	{
		Solid, /* Proxy for Blender's solid viewport shading */
		Normal, /* View object normals */
		Flat, /* Only see diffuse texture */
		Lines, /* Displays line list with color */
	};

	std::vector<std::shared_ptr<Object>> m_Objects; /* Objects to be drawn */

private:
	std::map<PipelineType, PipelineInfo> m_Pipelines; /* Pipelines with diff. shaders/draw modes */

	/* We set the camera's view and projection matrices as push constants for Vulkan rendering */
	struct PushConstants {
		alignas(16) glm::mat4 view = glm::mat4(1.0f);
		alignas(16) glm::mat4 proj = glm::mat4(1.0f);
	};

	VkPipelineLayout m_PipelineLayout;
	VkDescriptorSetLayout m_DescriptorSetLayout;
	VkDescriptorPool m_DescriptorPool;

	ImVec2 m_ViewportSize;
	VkSampleCountFlagBits m_MSAASampleCount;
	VkRenderPass m_ViewportRenderPass;
	std::vector<VkFramebuffer> m_ViewportFramebuffers;
};