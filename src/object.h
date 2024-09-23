#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "mesh.h"


class Object
{
public:
	Object(Mesh mesh, VkPipeline& pipeline);
	~Object();

	/* Adds draw command to the provided command buffer */
	void Draw(VkCommandBuffer& commandBuffer);

private:
	Mesh m_Mesh = Mesh();

	VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_IndexBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_VertexBufferMemory = VK_NULL_HANDLE;

	VkPipeline m_Pipeline = VK_NULL_HANDLE; /* The graphics pipeline to be used to draw this object */

	glm::mat4 m_Transform = glm::mat4(1.0f);
	glm::mat4 m_NormalTransform = glm::mat4(1.0f);
};