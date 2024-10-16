#pragma once

#include <map>

#include "vulkan/vulkan_utils.h"
#include "object.h"
#include "mesh.h"
#include "camera.h"
#include "application.h"
#include "common_enums.h"

/* Forward declarations */
class Object;
struct PipelineInfo;


class Scene
{
public:
	Scene(int scene);
	~Scene();

	void MakeScene(int scene);

	/* Utility functions for scene setup and rendering with Vulkan */
	void VkSetup(ImVec2 viewportSize, VkSampleCountFlagBits sampleCount, VkRenderPass& renderPass, std::vector<VkFramebuffer>& framebuffers);
	void VkResize(ImVec2 newSize, std::vector<VkFramebuffer>& framebuffers);
	void VkDraw(const Camera& camera);
	void VkCleanup();

public:
	enum SceneType
	{
		DEFAULT = 0,
		CORNELL_BOX,
		MAX_SCENE_COUNT
	};

	const std::map<int, std::string> m_SceneNames = {
		{DEFAULT, "Default"},
		{CORNELL_BOX, "Cornell Box"}
	};

	int m_BackgroundMode = BACKGROUND_MODE_SOLID_COLOR;

	/* Scene clear/background color used if m_BackgroundMode == SOLID_COLOR */
	glm::vec3 m_ClearColor = glm::vec3(63.0f / 255.0f, 63.0f / 255.0f, 63.0f / 255.0f);

	/* Scene bottom background gradient color used if m_BackgroundMode == GRADIENT */
	glm::vec3 m_GradientBottom = glm::vec3(0.3f);

	/* Scene top background gradient color used if m_BackgroundMode == GRADIENT */
	glm::vec3 m_GradientTop = glm::vec3(1.0f);

	/* Background texture loaded as a float to enable hdr skyboxes */
	Texture<float> m_BackgroundTexture;

	/* List of pipeline types for Vulkan rendering */
	enum PipelineType
	{
		VK_PIPELINE_SOLID, /* Proxy for Blender's solid viewport shading */
		VK_PIPELINE_NORMAL, /* View object normals */
		VK_PIPELINE_FLAT, /* Only see diffuse texture */
		VK_PIPELINE_LINES, /* Displays line list with color */
	};

	int m_SceneType;
	std::vector<std::shared_ptr<Material>> m_Materials;
	std::vector<std::shared_ptr<Mesh>> m_Meshes;
	std::vector<std::shared_ptr<Object>> m_RasterObjects; /* Objects to be drawn in RasterView */
	std::vector<std::shared_ptr<Object>> m_RayTraceObjects; /* Objects to be drawn in RayTraceView */

private:
	void PushToBoth(std::shared_ptr<Object> obj);

private:
	std::map<int, PipelineInfo> m_Pipelines; /* Vulkan Pipelines with diff. shaders/draw modes */

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