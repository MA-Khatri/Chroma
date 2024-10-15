#pragma once

#include <map>

#include "vulkan/vulkan_utils.h"
#include "object.h"
#include "mesh.h"
#include "camera.h"
#include "application.h"
#include "background_mode.h"

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
		DEFAULT,
		CORNELL_BOX
	};

	const std::map<int, std::string> m_SceneNames = {
		{DEFAULT, "Default"},
		{CORNELL_BOX, "Cornell Box"}
	};

	int m_BackgroundMode = BackgroundMode::SOLID_COLOR;

	/* Scene clear/background color used if m_BackgroundMode == SOLID_COLOR */
	glm::vec3 m_ClearColor = glm::vec3(63.0f / 255.0f, 63.0f / 255.0f, 63.0f / 255.0f);

	/* Scene bottom background gradient color used if m_BackgroundMode == GRADIENT */
	glm::vec3 m_GradientBottom = glm::vec3(0.3f);

	/* Scene top background gradient color used if m_BackgroundMode == GRADIENT */
	glm::vec3 m_GradientTop = glm::vec3(1.0f);

	/* Background texture loaded as a float to enable hdr skyboxes */
	Texture<float> m_BackgroundTexture;

	/* List of pipeline types for Vulkan rendering */
	enum PipelineType /* TODO: Should these be capitalized for consistency with other enums? */
	{
		Solid, /* Proxy for Blender's solid viewport shading */
		Normal, /* View object normals */
		Flat, /* Only see diffuse texture */
		Lines, /* Displays line list with color */
	};

	int m_SceneType = DEFAULT;
	std::vector<std::shared_ptr<Object>> m_RasterObjects; /* Objects to be drawn in RasterView */
	std::vector<std::shared_ptr<Object>> m_RayTraceObjects; /* Objects to be drawn in RayTraceView */

private:
	void PushToBoth(std::shared_ptr<Object> obj);

private:
	std::map<PipelineType, PipelineInfo> m_Pipelines; /* Vulkan Pipelines with diff. shaders/draw modes */

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