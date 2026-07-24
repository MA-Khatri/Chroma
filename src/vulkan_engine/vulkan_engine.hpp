#pragma once

#include "imgui.h"
#include "vk_scene.hpp"
#include "vulkan_utils.hpp"
#include <cstdint>
#include <vulkan/vulkan_core.h>

class VulkanEngine {
public:
  VulkanEngine();
  ~VulkanEngine();

  void OnResize(ImVec2 newSize);

  std::vector<uint32_t> SaveFramebuffer();

  const std::vector<VkDescriptorSet> &GetImageDescriptorSets() const {
    return m_ViewportImageDescriptorSets;
  }

  VkRenderPass GetRenderPass() const { return m_ViewportRenderPass; }

  void SetScene(std::shared_ptr<Scene> scene);

  // Draw call for main viewport render
  void DrawFrame();

  // Draw call for pick render pass (including depth buffer)
  void DrawPickFrame();

  std::tuple<VkRect2D, std::vector<float>> GetDepthBuffer(VkRect2D region);
  std::tuple<VkRect2D, std::vector<float>>
  GetDepthBuffer(int startX, int startY, unsigned int extentX, unsigned int extentY);
  std::tuple<VkRect2D, std::vector<float>> GetDepthBuffer();

  // Get depth buffer image of size m_PickDiameter * m_PickDiameter at provided center
  std::vector<float> GetPickDepth(int cx, int cy);

private:
  void InitVulkan();
  void CleanupVulkan();

  void CreateViewportImagesAndFramebuffers();
  void DestroyViewportImagesAndFramebuffers();

  void CreateViewportImageDescriptorSets();
  void DestroyViewportImageDescriptorSets();

  void DestroyColorResources();
  void DestroyDepthResources();

  // Pick-info frame resources
  void CreatePickResources();
  void DestroyPickResources();

  std::map<int, std::shared_ptr<VkScene>> m_VkScenes;
  std::shared_ptr<VkScene> m_VkScene; // Current active VkScene

  ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

  std::vector<VkImage> m_ViewportImages;
  std::vector<VkDeviceMemory> m_ViewportImagesDeviceMemory;
  std::vector<VkImageView> m_ViewportImageViews;
  std::vector<VkDescriptorSet> m_ViewportImageDescriptorSets;

  VkSampleCountFlagBits m_MSAASampleCount = VK_SAMPLE_COUNT_4_BIT;
  VkImage m_ColorImage; // for MSAA
  VkDeviceMemory m_ColorImageMemory;
  VkImageView m_ColorImageView;

  VkRenderPass m_ViewportRenderPass;
  VkPipelineLayout m_ViewportPipelineLayout;
  std::vector<VkFramebuffer> m_ViewportFramebuffers;
  VkSampler m_ViewportSampler;

  // Depth image for rendering
  VkImage m_DepthImage;
  VkDeviceMemory m_DepthImageMemory;
  VkImageView m_DepthImageView;

  // === Pick info frame resources ===
  bool m_AddPickRenderPass = false;
  VkRenderPass m_PickRenderPass;
  VkPipelineLayout m_PickPipelineLayout;
  VkFramebuffer m_PickFramebuffer;

  // Depth image for picking
  VkFormat m_PickDepthImageFormat;
  VkImage m_PickDepthImage;
  VkDeviceMemory m_PickDepthImageMemory;
  VkImageView m_PickDepthImageView;

  // Depth image for picking readback buffer
  static constexpr uint32_t m_PickDiameter = 5;
  static constexpr VkDeviceSize m_ReadbackSize = m_PickDiameter * m_PickDiameter * sizeof(float);
  VkBuffer m_PickDepthReadbackBuffer;
  VkDeviceMemory m_PickDepthReadbackMemory;
  void *m_PickDepthMappedReadback;

  // TODO: Add pick object and vertex id images?
};