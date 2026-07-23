#include "vk_object.hpp"
#include "vk_material.hpp"
#include <array>
#include <memory>

VkObject::VkObject(std::shared_ptr<Object> object, std::shared_ptr<VkMaterial> vkMaterial,
                   VkDescriptorSet sceneDescriptorSet)
    : m_Object(object), m_VkMaterial(vkMaterial), m_SceneDescriptorSet(sceneDescriptorSet) {
  // Create Vulkan buffers for the mesh
  vke::CreateVertexBuffer(m_Object->m_Mesh->vertices, m_VertexBuffer, m_VertexBufferMemory);
  vke::CreateIndexBuffer(m_Object->m_Mesh->indices, m_IndexBuffer, m_IndexBufferMemory);

  // Create object uniform buffer
  vke::CreateUniformBuffer(sizeof(ObjectUBO), m_UniformBuffer, m_UniformBufferMemory,
                           m_UniformBufferMapped);

  vke::CreateDescriptorSet(m_VkMaterial->m_PipelineInfo.objectDescriptorSetLayout,
                           m_VkMaterial->m_PipelineInfo.descriptorPool, m_DescriptorSet);

  VkUploadUniformBuffer();
  VkUpdateUniformBuffer();
}

VkObject::~VkObject() {
  vkUnmapMemory(vke::Device, m_UniformBufferMemory);
  vkDestroyBuffer(vke::Device, m_UniformBuffer, nullptr);
  vkFreeMemory(vke::Device, m_UniformBufferMemory, nullptr);

  vkDestroyBuffer(vke::Device, m_IndexBuffer, nullptr);
  vkFreeMemory(vke::Device, m_IndexBufferMemory, nullptr);

  vkDestroyBuffer(vke::Device, m_VertexBuffer, nullptr);
  vkFreeMemory(vke::Device, m_VertexBufferMemory, nullptr);
}

void VkObject::Draw(VkCommandBuffer commandBuffer) {
  // VkUploadUniformBuffer();

  std::array<VkDescriptorSet, 2> descriptorSets = {m_SceneDescriptorSet, m_DescriptorSet};
  vkCmdBindDescriptorSets(
      commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VkMaterial->m_PipelineInfo.pipelineLayout,
      0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_VkMaterial->m_PipelineInfo.pipeline);

  // Bind vertex and index buffers
  VkBuffer vertexBuffers[] = {m_VertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
  vkCmdBindIndexBuffer(commandBuffer, m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

  vkCmdSetLineWidth(commandBuffer, m_Object->m_Material->m_LineWidth);

  // Draw indexed
  vkCmdDrawIndexed(commandBuffer,
                   static_cast<uint32_t>(m_Object->m_Mesh->indices.size()), // indexCount
                   1,                                                       // instanceCount
                   0,                                                       // firstIndex
                   0,                                                       // vertexOffset
                   0);                                                      // firstInstance
}

void VkObject::DrawPick(VkCommandBuffer commandBuffer) {

  std::array<VkDescriptorSet, 2> descriptorSets = {m_SceneDescriptorSet, m_DescriptorSet};
  vkCmdBindDescriptorSets(
      commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VkMaterial->m_PipelineInfo.pipelineLayout,
      0, static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_VkMaterial->m_PipelineInfo.pipeline);

  // Bind vertex and index buffers
  VkBuffer vertexBuffers[] = {m_VertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
  vkCmdBindIndexBuffer(commandBuffer, m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

  vkCmdSetLineWidth(commandBuffer, m_Object->m_Material->m_LineWidth);

  // Draw indexed
  vkCmdDrawIndexed(commandBuffer,
                   static_cast<uint32_t>(m_Object->m_Mesh->indices.size()), // indexCount
                   1,                                                       // instanceCount
                   0,                                                       // firstIndex
                   0,                                                       // vertexOffset
                   0);                                                      // firstInstance
}

void VkObject::VkUpdateUniformBuffer() {
  std::vector<VkWriteDescriptorSet> descriptorWrites;
  descriptorWrites.reserve(m_VkMaterial->m_DescriptorWrites.size() + 1);

  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = m_UniformBuffer;
  bufferInfo.offset = 0;
  bufferInfo.range = sizeof(ObjectUBO);

  VkWriteDescriptorSet uboWrite{};
  uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  uboWrite.dstSet = m_DescriptorSet;
  uboWrite.dstBinding = 0;
  uboWrite.dstArrayElement = 0;
  uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboWrite.descriptorCount = 1;
  uboWrite.pBufferInfo = &bufferInfo;
  uboWrite.pImageInfo = nullptr;       // optional
  uboWrite.pTexelBufferView = nullptr; // optional
  descriptorWrites.push_back(uboWrite);

  for (const auto &write : m_VkMaterial->m_DescriptorWrites) {
    VkWriteDescriptorSet objectWrite = write;
    objectWrite.dstSet = m_DescriptorSet;
    descriptorWrites.push_back(objectWrite);
  }

  vkUpdateDescriptorSets(vke::Device, static_cast<uint32_t>(descriptorWrites.size()),
                         descriptorWrites.data(), 0, nullptr);
}

void VkObject::VkUploadUniformBuffer() {
  if (m_UniformBufferMapped) {
    ObjectUBO ubo{};
    ubo.modelMatrix = m_Object->m_Transform->GetModelMatrix();
    ubo.normalMatrix = glm::mat4(m_Object->m_Transform->GetNormalMatrix());
    memcpy(m_UniformBufferMapped, &ubo, sizeof(ubo));
  } else {
    PLOG_WARNING << "m_UniformBufferMapped is nullptr!" << std::endl;
  }
}