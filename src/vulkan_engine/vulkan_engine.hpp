#pragma once

#include "vulkan_utils.hpp"
#include "imgui.h"

class VulkanEngine {
public:
  VulkanEngine();
  ~VulkanEngine();

  void OnResize(ImVec2 newSize);

  std::vector<uint32_t> SaveFramebuffer();

  // TODO: Add scene rendering methods
  // void RenderScene(Camera &camera, Scene &scene);

  const std::vector<VkDescriptorSet> &GetImageDescriptorSets() const {
    return m_ViewportImageDescriptorSets;
  }

private:
  void InitVulkan();
  void CleanupVulkan();
  void CreateViewportImagesAndFramebuffers();
  void DestroyViewportImagesAndFramebuffers();
  void CreateViewportImageDescriptorSets();
  void DestroyViewportImageDescriptorSets();
  void DestroyColorResources();
  void DestroyDepthResources();

  ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

  std::vector<VkImage> m_ViewportImages;
  std::vector<VkDeviceMemory> m_ViewportImagesDeviceMemory;
  std::vector<VkImageView> m_ViewportImageViews;
  std::vector<VkDescriptorSet> m_ViewportImageDescriptorSets;

  VkSampleCountFlagBits m_MSAASampleCount = VK_SAMPLE_COUNT_1_BIT;
  VkImage m_ColorImage; // for MSAA
  VkDeviceMemory m_ColorImageMemory;
  VkImageView m_ColorImageView;

  VkRenderPass m_ViewportRenderPass;
  VkPipelineLayout m_ViewportPipelineLayout;
  std::vector<VkFramebuffer> m_ViewportFramebuffers;
  VkSampler m_ViewportSampler;

  VkImage m_DepthImage;
  VkDeviceMemory m_DepthImageMemory;
  VkImageView m_DepthImageView;
};