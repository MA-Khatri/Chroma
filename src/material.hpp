#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3_image/SDL_image.h>
#include <glm/glm.hpp>
#include <plog/Log.h>
#include <vulkan/vulkan.h>

struct PipelineInfo {
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};

struct TexturePaths {
  std::string diffuse;
  std::string specular;
  std::string normal;
};

// Local texture storage
template <typename T> struct Texture {
  std::string filePath;  // path to texture image
  std::vector<T> pixels; // local storage of pixels
  glm::ivec3 resolution; // x = width, y = height, z = channels
  int textureID = -1;    // textureID set in optix_renderer -> CreateTextures()

  void LoadTexture() {
    SDL_Surface *imageSurface = IMG_Load(filePath.c_str());
    if (!imageSurface) {
      PLOG_ERROR << "LoadTexture(): Error! Failed to load image " << filePath
                 << "! SDL_Image Error: " << SDL_GetError();
      return;
    }

    // Convert surface to RGBA32 format
    SDL_Surface *convertedSurface =
        SDL_ConvertSurface(imageSurface, SDL_PIXELFORMAT_RGBA32);
    SDL_DestroySurface(imageSurface); // Free original surface
    if (!convertedSurface) {
      PLOG_ERROR
          << "LoadTexture(): Error! Failed to convert image to RGBA32 format: "
          << SDL_GetError();
      return;
    }

    int texWidth = convertedSurface->w;
    int texHeight = convertedSurface->h;
    int texChannels = 4;
    resolution = glm::ivec3(texWidth, texHeight, texChannels);
    size_t dataSize = texWidth * texHeight * texChannels;

    if (std::is_same<T, uint8_t>::value) {
      uint8_t *data = static_cast<uint8_t *>(convertedSurface->pixels);
      pixels.assign(data, data + dataSize);
    } else if (std::is_same<T, float>::value) {
      // Convert uint8_t to float
      uint8_t *data = static_cast<uint8_t *>(convertedSurface->pixels);
      pixels.resize(dataSize);
      for (size_t i = 0; i < dataSize; ++i) {
        pixels[i] = static_cast<float>(data[i]) / 255.0f;
      }
    } else {
      PLOG_ERROR << "LoadTexture(): Error! Unsupported texture format!";
      SDL_DestroySurface(convertedSurface);
      return;
    }

    SDL_DestroySurface(convertedSurface);
    return;
  }
};

class Material {
public:
  Material(TexturePaths texturePaths, int vkPipelineType, int rtMaterialType);
  ~Material();

  void LoadTextures();

  // Sets up material to be drawn with Vulkan
  void VkSetup(const PipelineInfo &pipelineInfo);

public:
  bool m_DepthTest = true;

  TexturePaths m_TexturePaths;
  Texture<uint8_t> m_DiffuseTexture;
  Texture<uint8_t> m_SpecularTexture;
  Texture<uint8_t> m_NormalTexture;

  // Ray tracing material type -- i.e., MaterialType enum
  int m_RTMaterialType = 0;

  // Material properties
  float m_Roughness = 0.0f;
  float m_EtaIn = 1.0f;
  float m_EtaOut = 1.0f;
  glm::vec3 m_ReflectionColor = glm::vec3(1.0f);
  glm::vec3 m_RefractionColor = glm::vec3(1.0f);
  glm::vec3 m_Extinction = glm::vec3(0.0f);
  // I.e., radiant exitance -- emitted flux per unit area
  glm::vec3 m_EmissionColor = glm::vec3(0.0f);

  // The vulkan graphics pipeline to be used to draw this material
  int m_VKPipelineType = -1; // Used to access the Scene::PipelineType enum
  VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
  VkPipeline m_Pipeline = VK_NULL_HANDLE;
  VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
  std::vector<VkWriteDescriptorSet> m_DescriptorWrites;

private:
  // Textures
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