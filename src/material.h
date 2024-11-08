#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <string>
#include <iostream>
#include <vector>

#include "stb_image.h"


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
template<typename T>
struct Texture
{
	std::string filePath; /* path to texture image */
	std::vector<T> pixels; /* local storage of pixels */
	glm::ivec3 resolution; /* x = width, y = height, z = channels */
	int textureID = -1; /* textureID set in optix_renderer -> CreateTextures() */

	void LoadTexture()
	{
		int texWidth, texHeight, texChannels;
		T* data;
		if (std::is_same<T, uint8_t>::value)
		{
			/* Note: we load image with alpha channel even if it doesn't have one */
			data = (T*)stbi_load(filePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		}
		else if (std::is_same<T, float>::value)
		{
			data = (T*)stbi_loadf(filePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		}
		else
		{
			std::cerr << "LoadTexture(): Error! Unsupported texture format!" << std::endl;
		}

		if (!data)
		{
			std::cerr << "LoadTexture(): Error! Failed to load image " << filePath << " ! " << std::endl;
			exit(-1);
		}

#ifdef _DEBUG
		if (texChannels != 4) std::cout << "Warning: loaded texture only has " << texChannels << " channels. Force loaded with 4 channels!" << std::endl;
#endif
		resolution = glm::ivec3(texWidth, texHeight, 4);

		/* Copy pixels to local std::vector and free originally read data */
		pixels = std::vector<T>(data, data + (resolution.x * resolution.y * resolution.z));
		stbi_image_free(data);
	}
};


class Material
{
public:
	Material(TexturePaths texturePaths, int vkPipelineType, int rtMaterialType);
	~Material();

	void LoadTextures();

	/* Sets up material to be drawn with Vulkan */
	void VkSetup(const PipelineInfo& pipelineInfo);

public:
	bool m_DepthTest = true;

	TexturePaths m_TexturePaths;
	Texture<uint8_t> m_DiffuseTexture;
	Texture<uint8_t> m_SpecularTexture;
	Texture<uint8_t> m_NormalTexture;

	int m_RTMaterialType = 0; /* Ray tracing material type -- i.e., MaterialType enum */

	/* Material properties */
	float m_Roughness = 0.0f;
	float m_EtaIn = 1.0f;
	float m_EtaOut = 1.0f;
	glm::vec3 m_ReflectionColor = glm::vec3(1.0f);
	glm::vec3 m_RefractionColor = glm::vec3(1.0f);
	glm::vec3 m_Extinction = glm::vec3(0.0f);
	glm::vec3 m_EmissionColor = glm::vec3(0.0f); /* I.e., radiant exitance -- emitted flux per unit area */

	/* The vulkan graphics pipeline to be used to draw this material */
	int m_VKPipelineType = -1; /* Used to access the Scene::PipelineType enum */
	VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
	VkPipeline m_Pipeline = VK_NULL_HANDLE;
	VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
	std::vector<VkWriteDescriptorSet> m_DescriptorWrites;

private:
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