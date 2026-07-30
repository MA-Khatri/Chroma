#pragma once

#include "../scene.hpp"
#include "imgui.h"
#include "vk_object.hpp"
#include "vulkan_utils.hpp"

#include <map>
#include <memory>
#include <vulkan/vulkan_core.h>

class VulkanEngine; // Forward declaration

class VkScene {
public:
  VkScene(std::shared_ptr<Scene> scene, VkSampleCountFlagBits msaaCount, VkRenderPass renderPass,
          VkRenderPass pickRenderPass);
  ~VkScene();

  struct alignas(16) SceneUBO {
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
    glm::mat4 viewProjectionMatrix;
    alignas(16) glm::vec3 cameraPosition;
    alignas(16) glm::vec2 viewportSize;
  };

  void Draw(VkCommandBuffer commandBuffer, ImVec2 viewportSize);

  void DrawPick(VkCommandBuffer commandBuffer, ImVec2 viewportSize);

  std::shared_ptr<Scene> GetBaseScene() const { return m_Scene; }

  void ReplaceObject(int idx, std::shared_ptr<Object> object) {
    if (idx < 0 || idx >= m_VkObjects.size()) {
      PLOG_ERROR << "Invalid object index to replace!";
      return;
    }

    if (m_DescriptorSet == VK_NULL_HANDLE) {
      PLOG_ERROR << "Invalid descriptor set";
      return;
    }

    PLOG_INFO << "0";

    auto it = m_Materials.find(object->m_Material->m_MaterialID);
    if (it == m_Materials.end()) {
      PLOG_ERROR << "No material found for ID " << object->m_Material->m_MaterialID;
      return;
    }
    auto material = it->second;

    if (material == VK_NULL_HANDLE) {
      PLOG_ERROR << "Material is null!";
    }

    PLOG_INFO << "1";

    auto newVkObject = std::make_shared<VkObject>(object, material, m_DescriptorSet);

    PLOG_INFO << "2";

    m_VkObjects[idx] = newVkObject;

    PLOG_INFO << "3";
  }

private:
  // Updates descriptor set to take in this scene's uniform buffer
  void VkUpdateUniformBuffer();

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