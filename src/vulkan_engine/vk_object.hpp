#pragma once

#include "../object.hpp"
#include "vk_material.hpp"
#include "vulkan_utils.hpp"
#include <memory>

class VkObject {
public:
  VkObject(std::shared_ptr<Object> object, std::shared_ptr<VkMaterial> material);
  ~VkObject();

  struct UniformBufferObject {
    glm::mat4 modelMatrix;
    glm::mat4 normalMatrix;
  };

  void Draw(VkCommandBuffer commandBuffer);

private:
  void VkUpdateUniformBuffer(); // Updates descriptor set to take in this object's uniform buffer
  void VkUploadUniformBuffer(); // Uploads the uniform buffer data to the GPU

  std::shared_ptr<Object> m_Object; // The original object

  std::shared_ptr<VkMaterial> m_VkMaterial;

  VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_VertexBufferMemory = VK_NULL_HANDLE;

  VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_IndexBufferMemory = VK_NULL_HANDLE;

  // Uniform buffer
  VkBuffer m_UniformBuffer = VK_NULL_HANDLE;
  VkDeviceMemory m_UniformBufferMemory = VK_NULL_HANDLE;
  void *m_UniformBufferMapped = nullptr;
};