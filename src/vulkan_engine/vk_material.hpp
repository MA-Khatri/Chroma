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
  VkPipeline pickPipeline = VK_NULL_HANDLE;
};

class VkMaterial {
public:
  VkMaterial(std::shared_ptr<Material> material, VkDescriptorPool descriptorPool,
             VkDescriptorSetLayout sceneDescriptorSetLayout, VkSampleCountFlagBits msaaCount,
             VkRenderPass renderPass, VkRenderPass pickRenderPass = VK_NULL_HANDLE);
  ~VkMaterial();

  // The vulkan graphics pipeline to be used to draw this material
  PipelineInfo m_PipelineInfo;

  std::vector<VkWriteDescriptorSet> m_DescriptorWrites;

private:
  std::array<VkDescriptorImageInfo, 3> m_DescriptorImageInfos{};
  uint32_t m_DescriptorImageInfoCount = 0;

  // Textures
  VkImage m_AlbedoTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory m_AlbedoTextureImageMemory = VK_NULL_HANDLE;
  VkImageView m_AlbedoTextureImageView = VK_NULL_HANDLE;
  VkSampler m_AlbedoTextureSampler = VK_NULL_HANDLE;
  uint32_t m_AlbedoMipLevels = 0;

  // VkImage m_RoughnessTextureImage = VK_NULL_HANDLE;
  // VkDeviceMemory m_RoughnessTextureImageMemory = VK_NULL_HANDLE;
  // VkImageView m_RoughnessTextureImageView = VK_NULL_HANDLE;
  // VkSampler m_RoughnessTextureSampler = VK_NULL_HANDLE;
  // uint32_t m_RoughnessMipLevels = 0;

  // VkImage m_NormalTextureImage = VK_NULL_HANDLE;
  // VkDeviceMemory m_NormalTextureImageMemory = VK_NULL_HANDLE;
  // VkImageView m_NormalTextureImageView = VK_NULL_HANDLE;
  // VkSampler m_NormalTextureSampler = VK_NULL_HANDLE;
  // uint32_t m_NormalMipLevels = 0;
};