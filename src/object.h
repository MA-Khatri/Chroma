#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <vulkan/vulkan.h>

#include "mesh.h"


class Object
{
public:
	Object(Mesh mesh, VkPipeline& pipeline);
	~Object();

	/* Adds draw command to the provided command buffer */
	void Draw(VkCommandBuffer& commandBuffer);

	/* === Transformations === */
	void UpdateModelNormalMatrix();
	void SetModelMatrix(glm::mat4x4 matrix);
	void UpdateModelMatrix(glm::mat4x4 matrix);

	void Translate(glm::vec3 translate);
	void Translate(float x, float y, float z);

	void Rotate(glm::vec3 axis, float deg);

	void Scale(glm::vec3 scale);
	void Scale(float scale);
	void Scale(float x, float y, float z);


public:
	glm::mat4 m_ModelMatrix = glm::mat4(1.0f);
	glm::mat4 m_ModelNormalMatrix = glm::mat4(1.0f);


private:
	Mesh m_Mesh = Mesh();

	VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_IndexBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_VertexBufferMemory = VK_NULL_HANDLE;

	VkPipeline m_Pipeline = VK_NULL_HANDLE; /* The graphics pipeline to be used to draw this object */
};