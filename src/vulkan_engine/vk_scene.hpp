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

  struct alignas(16) SceneUBO {
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    glm::mat4 viewProjectionMatrix;
    glm::vec4 cameraPositionAndViewportHeight; // [pos.x, pos.y, pos.z, viewportHeight]
  };

  void Draw(VkCommandBuffer commandBuffer, ImVec2 viewportSize);

  std::shared_ptr<Scene> GetBaseScene() const { return m_Scene; }

private:
  // Uploads the uniform buffer data to the GPU
  void VkUploadUniformBuffer(ImVec2 viewportSize);

  std::shared_ptr<Scene> m_Scene; // Original scene

  std::vector<std::shared_ptr<VkObject>> m_VkObjects;
  std::map<int, std::shared_ptr<VkMaterial>> m_Materials;

  VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;

  // Uniform buffer for scene-wide uniforms
  VkBuffer m_UniformBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_UniformBufferMemory = VK_NULL_HANDLE;
  void *m_UniformBufferMapped = nullptr;
  VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
};