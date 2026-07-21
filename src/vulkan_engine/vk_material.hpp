#pragma once

#include <array>
#include <memory>
#include <vector>

#include "../material.hpp"
#include "vulkan_utils.hpp"
#include <vulkan/vulkan_core.h>

struct PipelineInfo {
  VkDescriptorSetLayout objectDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};

struct PushConstants {
  alignas(16) glm::mat4 view = glm::mat4(1.0f);
  alignas(16) glm::mat4 proj = glm::mat4(1.0f);
};

class VkMaterial {
public:
  VkMaterial(std::shared_ptr<Material> material, VkDescriptorPool descriptorPool,
             VkSampleCountFlagBits msaaCount, VkRenderPass renderPass);
  ~VkMaterial();

  // The vulkan graphics pipeline to be used to draw this material
  PipelineInfo m_PipelineInfo;

  std::vector<VkWriteDescriptorSet> m_DescriptorWrites;

private:
  std::array<VkDescriptorImageInfo, 3> m_DescriptorImageInfos{};
  uint32_t m_DescriptorImageInfoCount = 0;

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