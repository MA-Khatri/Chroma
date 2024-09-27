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


class Object
{
public:
	Object(Mesh mesh, VkDescriptorSetLayout& descriptorSetLayout, VkDescriptorPool& descriptorPool, VkPipelineLayout& pipelineLayout, VkPipeline& pipeline);
	~Object();

	/* Adds draw command to the provided command buffer */
	void Draw(VkCommandBuffer& commandBuffer);

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


private:
	void UpdateUniformBuffer();


public:
	glm::mat4 m_ModelMatrix = glm::mat4(1.0f);
	glm::mat3 m_ModelNormalMatrix = glm::mat3(1.0f);

	/* Uniform buffer contains all necessary drawing info for this object */
	struct UniformBufferObject {
		alignas(16) glm::mat4 modelMatrix;
		alignas(16) glm::mat3 normalMatrix;
	};


private:
	Mesh m_Mesh = Mesh();

	VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_VertexBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_IndexBufferMemory = VK_NULL_HANDLE;

	/* The graphics pipeline to be used to draw this object */
	VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
	VkPipeline m_Pipeline = VK_NULL_HANDLE;

	VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> m_DescriptorSets;

	std::vector<VkBuffer> m_UniformBuffers;
	std::vector<VkDeviceMemory> m_UniformBuffersMemory;
	std::vector<void*> m_UniformBuffersMapped;

	VkImage m_TextureImage;
	VkDeviceMemory m_TextureImageMemory;
	VkImageView m_TextureImageView;
	VkSampler m_TextureSampler;
};