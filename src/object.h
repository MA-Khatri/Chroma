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


struct PipelineInfo
{
	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
};

struct TexturePaths
{
	std::string diffuse;
	std::string specular;
	std::string normal;
};

/* Local texture storage */
struct Texture
{
	std::string filePath;
	std::vector<uint8_t> pixels; /* local storage of pixels */
	glm::ivec3 resolution; /* x = width, y = height, z = channels */
	int textureID = -1; /* textureID set in optix_renderer -> CreateTextures() */
};


class Object
{
public:
	Object(Mesh mesh, TexturePaths texturePaths, int pipelineType);
	~Object();

	/* Load in the pixels and resolution of the provided texture */
	void LoadTexture(Texture& tex);

	/* Sets up object to be drawn with Vulkan */
	void VkSetup(const PipelineInfo& pipelineInfo);

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
	bool m_DepthTest = true;
	int m_PipelineType = 0; /* Used to access the Scene::PipelineType enum */

	/* Uniform buffer contains all necessary drawing info for this object */
	struct UniformBufferObject {
		alignas(16) glm::mat4 modelMatrix;
		alignas(16) glm::mat4 normalMatrix; /* We pass in the normal matrix as a mat4 to avoid alignment issues */
	};

	Mesh m_Mesh = Mesh();
	TexturePaths m_TexturePaths;
	Texture m_DiffuseTexture;
	Texture m_SpecularTexture;
	Texture m_NormalTexture;

private:
	/* === Vulkan === */
	VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_VertexBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_IndexBufferMemory = VK_NULL_HANDLE;

	/* The graphics pipeline to be used to draw this object */
	VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
	VkPipeline m_Pipeline = VK_NULL_HANDLE;

	/* Descriptor set info */
	VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;

	/* Uniform buffer */
	VkBuffer m_UniformBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_UniformBufferMemory = VK_NULL_HANDLE;
	void* m_UniformBufferMapped = nullptr;

	/* Textures */
	VkImage m_DiffuseTextureImage = VK_NULL_HANDLE;
	VkDeviceMemory m_DiffuseTextureImageMemory = VK_NULL_HANDLE;
	VkImageView m_DiffuseTextureImageView = VK_NULL_HANDLE;
	VkSampler m_DiffuseTextureSampler = VK_NULL_HANDLE;
	uint32_t m_DiffuseMipLevels = 0;

	VkImage m_SpecularTextureImage = VK_NULL_HANDLE;
	VkDeviceMemory m_SpecularTextureImageMemory = VK_NULL_HANDLE;
	VkImageView m_SpecularTextureImageView = VK_NULL_HANDLE;
	VkSampler m_SpecularTextureSampler = VK_NULL_HANDLE;
	uint32_t m_SpecularMipLevels = 0;

	VkImage m_NormalTextureImage = VK_NULL_HANDLE;
	VkDeviceMemory m_NormalTextureImageMemory = VK_NULL_HANDLE;
	VkImageView m_NormalTextureImageView = VK_NULL_HANDLE;
	VkSampler m_NormalTextureSampler = VK_NULL_HANDLE;
	uint32_t m_NormalMipLevels = 0;
};