#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <vulkan/vulkan.h>
#include <vector>

#include "mesh.h"
#include "material.h"


class Object
{
public:
	Object(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material);
	~Object();


	void VkSetup();

	/* Adds draw command to the provided command buffer */
	void VkDraw(VkCommandBuffer& commandBuffer);

	/* === Transformations === */
	void UpdateModelNormalMatrix();
	void SetModelMatrix(glm::mat4 matrix);
	void UpdateModelMatrix(glm::mat4 matrix);

	void Translate(glm::vec3 translate);
	void Translate(float x, float y, float z);

	void Rotate(glm::vec3 axis, float deg);

	void Scale(glm::vec3 scale, bool updateNormal = true);
	void Scale(float scale);
	void Scale(float x, float y, float z);

	void VkUpdateUniformBuffer();

public:
	glm::mat4 m_ModelMatrix = glm::mat4(1.0f);
	glm::mat3 m_ModelNormalMatrix = glm::mat3(1.0f);

	/* Uniform buffer contains all necessary drawing info for this object */
	struct UniformBufferObject {
		alignas(16) glm::mat4 modelMatrix;
		alignas(16) glm::mat4 normalMatrix; /* We pass in the normal matrix as a mat4 to avoid alignment issues */

		alignas(16) glm::vec3 color;
	};

	std::shared_ptr<Mesh> m_Mesh;
	std::shared_ptr<Material> m_Material;

private:
	/* === Vulkan === */
	VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_VertexBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_IndexBufferMemory = VK_NULL_HANDLE;

	/* Uniform buffer */
	VkBuffer m_UniformBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_UniformBufferMemory = VK_NULL_HANDLE;
	void* m_UniformBufferMapped = nullptr;

	std::vector<VkWriteDescriptorSet> m_DescriptorWrites;
};