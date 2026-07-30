#include "vk_scene.hpp"
#include "vk_material.hpp"
#include "vulkan_utils.hpp"
#include <exception>
#include <map>
#include <set>
#include <vulkan/vulkan_core.h>

VkScene::VkScene(std::shared_ptr<Scene> scene, VkSampleCountFlagBits msaaCount,
                 VkRenderPass renderPass, VkRenderPass pickRenderPass)
    : m_Scene(scene) {
  PLOG_DEBUG << "Creating VkScene for Scene ID: " << m_Scene->m_SceneID;

  // Create descriptor pool
  std::set<int> materialIDs;
  for (const auto &object : m_Scene->GetObjects()) {
    materialIDs.insert(object->m_Material->m_MaterialID);
  }
  vke::CreateDescriptorPool(materialIDs.size(), m_Scene->GetObjects().size() + 100000,
                            m_DescriptorPool);

  // Create scene uniform buffer and its associated descriptor set
  vke::CreateUniformBuffer(sizeof(SceneUBO), m_UniformBuffer, m_UniformBufferMemory,
                           m_UniformBufferMapped);

  std::vector<VkDescriptorSetLayoutBinding> sceneBindings;
  VkDescriptorSetLayoutBinding sceneBinding{};
  sceneBinding.binding = 0;
  sceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sceneBinding.descriptorCount = 1;
  sceneBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  sceneBinding.pImmutableSamplers = nullptr;
  sceneBindings.push_back(sceneBinding);

  vke::CreateDescriptorSetLayout(sceneBindings, m_DescriptorSetLayout);
  vke::CreateDescriptorSet(m_DescriptorSetLayout, m_DescriptorPool, m_DescriptorSet);

  VkUpdateUniformBuffer();
  VkUploadUniformBuffer(ImVec2(1, 1));

  // Initialize unique materials
  for (const auto &object : m_Scene->GetObjects()) {
    int matID = object->m_Material->m_MaterialID;
    if (m_Materials.find(matID) == m_Materials.end()) {
      m_Materials.insert({matID, std::make_shared<VkMaterial>(object->m_Material, m_DescriptorPool,
                                                              m_DescriptorSetLayout, msaaCount,
                                                              renderPass, pickRenderPass)});
    }
  }

  // For pipeline layout, we can just use the layout from any material
  // -- this will only be used to set the push constants
  if (m_Materials.size() > 0) {
    auto mat = m_Materials.begin()->second;
    m_PipelineLayout = mat->m_PipelineInfo.pipelineLayout;
  } else {
    PLOG_WARNING << "Missing materials for VkScene!";
  }

  // Create VkObject for each Object in the Scene
  for (const auto &object : m_Scene->GetObjects()) {
    auto material = m_Materials[object->m_Material->m_MaterialID];
    m_VkObjects.push_back(std::make_shared<VkObject>(object, material, m_DescriptorSet));
  }

  PLOG_DEBUG << "Done creating VkScene for Scene ID: " << m_Scene->m_SceneID;
}

VkScene::~VkScene() {
  vkDestroyDescriptorPool(vke::Device, m_DescriptorPool, nullptr);
  vkDestroyPipelineLayout(vke::Device, m_PipelineLayout, nullptr);
}

void VkScene::Draw(VkCommandBuffer commandBuffer, ImVec2 viewportSize) {
  // Upload scene UBO
  VkUploadUniformBuffer(viewportSize);

  for (const auto &vkObject : m_VkObjects) {
    vkObject->Draw(commandBuffer, viewportSize);
  }
}

void VkScene::DrawPick(VkCommandBuffer commandBuffer, ImVec2 viewportSize) {
  // Upload scene UBO
  VkUploadUniformBuffer(viewportSize);

  for (const auto &vkObject : m_VkObjects) {
    vkObject->DrawPick(commandBuffer, viewportSize);
  }
}

void VkScene::ReplaceObject(int idx, std::shared_ptr<Object> object) {
  if (idx < 0 || idx >= m_VkObjects.size()) {
    PLOG_ERROR << "Invalid object index to replace!";
    return;
  }

  if (m_DescriptorSet == VK_NULL_HANDLE) {
    PLOG_ERROR << "Invalid descriptor set";
    return;
  }

  if (!object->m_Mesh) {
    PLOG_ERROR << "Object mesh is nullptr!";
    return;
  }

  int materialID = object->m_Material->m_MaterialID;
  auto it = m_Materials.find(materialID);
  if (it == m_Materials.end()) {
    PLOG_ERROR << "No material found for ID " << materialID;
    return;
  }
  std::shared_ptr<VkMaterial> material = it->second;

  if (material == nullptr) {
    PLOG_ERROR << "Material is null!";
    return;
  }

  m_VkObjects[idx] = std::make_shared<VkObject>(object, material, m_DescriptorSet);
}

void VkScene::VkUpdateUniformBuffer() {
  VkDescriptorBufferInfo sceneBufferInfo{};
  sceneBufferInfo.buffer = m_UniformBuffer;
  sceneBufferInfo.offset = 0;
  sceneBufferInfo.range = sizeof(SceneUBO);

  VkWriteDescriptorSet sceneWrite{};
  sceneWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  sceneWrite.dstSet = m_DescriptorSet;
  sceneWrite.dstBinding = 0;
  sceneWrite.dstArrayElement = 0;
  sceneWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  sceneWrite.descriptorCount = 1;
  sceneWrite.pBufferInfo = &sceneBufferInfo;

  vkUpdateDescriptorSets(vke::Device, 1, &sceneWrite, 0, nullptr);
}

void VkScene::VkUploadUniformBuffer(ImVec2 viewportSize) {
  if (m_UniformBufferMapped) {
    auto camera = m_Scene->GetCamera();
    if (!camera) {
      PLOG_WARNING << "Scene does not have a camera to upload!";
      return;
    }

    SceneUBO ubo{};
    ubo.viewMatrix = camera->GetViewMatrix();
    ubo.projectionMatrix = camera->GetProjectionMatrix();
    ubo.viewProjectionMatrix = camera->GetViewProjectionMatrix();
    ubo.cameraPosition = camera->GetPosition();
    ubo.viewportSize = glm::vec2(viewportSize.x, viewportSize.y);

    memcpy(m_UniformBufferMapped, &ubo, sizeof(ubo));
  } else {
    PLOG_WARNING << "m_UniformBufferMapped is nullptr!" << std::endl;
  }
}
