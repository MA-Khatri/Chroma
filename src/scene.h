#pragma once

#include <map>

#include "vulkan_utils.h"
#include "object.h"
#include "mesh.h"
#include "camera.h"
#include "application.h"

class Scene
{
public:
	Scene() {}; /* Bogus initializer to prevent errors */
	Scene(ImVec2& viewportSize, VkSampleCountFlagBits& sampleCount, VkRenderPass& renderPass, std::vector<VkFramebuffer>* framebuffers, Application* app, Camera* camera);
	~Scene();

	void Setup();
	void VkDraw();

private:
	/* List of pipeline types */
	enum Pipelines
	{
		Flat, /* Only see diffuse texture */
		Normal, /* View object normals */
		Solid, /* Proxy for Blender's solid viewport shading */
		Lines, /* Displays line list with color */
	};

	std::map<Pipelines, PipelineInfo> m_Pipelines; /* Pipelines with diff. shaders/draw modes */

	std::vector<Object*> m_Objects; /* Objects to be drawn */

	/* We set the camera's view and projection matrices as push constants */
	struct PushConstants {
		alignas(16) glm::mat4 view = glm::mat4(1.0f);
		alignas(16) glm::mat4 proj = glm::mat4(1.0f);
	};

	VkPipelineLayout m_PipelineLayout;
	VkDescriptorSetLayout m_DescriptorSetLayout;
	VkDescriptorPool m_DescriptorPool;

	/* Initialized in constructor by the rasterized layer */
	ImVec2 m_ViewportSize;
	VkSampleCountFlagBits m_MSAASampleCount;
	VkRenderPass m_ViewportRenderPass;
	std::vector<VkFramebuffer>* m_ViewportFramebuffers;
	Application* m_AppHandle;
	Camera* m_Camera;
};