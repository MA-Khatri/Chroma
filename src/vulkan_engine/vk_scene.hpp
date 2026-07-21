#pragma once

#include "../scene.hpp"
#include "imgui.h"
#include "vk_object.hpp"

#include <map>
#include <vulkan/vulkan_core.h>

class VulkanEngine; // Forward declaration

class VkScene {
public:
  VkScene(std::shared_ptr<Scene> scene, VkSampleCountFlagBits msaaCount, VkRenderPass renderPass);
  ~VkScene();

  void Draw(VkCommandBuffer commandBuffer);

  std::shared_ptr<Scene> GetBaseScene() const { return m_Scene; }

private:
  std::shared_ptr<Scene> m_Scene; // Original scene

  std::vector<std::shared_ptr<VkObject>> m_VkObjects;
  std::map<int, std::shared_ptr<VkMaterial>> m_Materials;

  VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
};