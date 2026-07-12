#include "vk_scene.hpp"
#include <map>
#include <set>
#include <vulkan/vulkan_core.h>

VkScene::VkScene(std::shared_ptr<Scene> scene, ImVec2 viewportSize, VkSampleCountFlagBits msaaCount,
                 VkRenderPass renderPass)
    : m_Scene(scene) {
  PLOG_DEBUG << "Creating VkScene for Scene ID: " << m_Scene->m_SceneID;

  // Create descriptor pool
  std::set<int> materialIDs;
  for (const auto &object : m_Scene->GetObjects()) {
    materialIDs.insert(object->m_Material->m_MaterialID);
  }
  const uint32_t descriptorSetCount =
      static_cast<uint32_t>(materialIDs.size() + m_Scene->GetObjects().size());
  vke::CreateDescriptorPool(descriptorSetCount, m_DescriptorPool);

  // Create a list of all materials used in the scene
  for (const auto &object : m_Scene->GetObjects()) {
    int matID = object->m_Material->m_MaterialID;
    if (m_Materials.find(matID) == m_Materials.end()) {
      m_Materials.insert(
          {matID, std::make_shared<VkMaterial>(object->m_Material, m_DescriptorPool, viewportSize,
                                               msaaCount, renderPass)});
    }
  }

  // For pipeline layout, we can just use the layout from any material
  // -- this will only be used to set the push constants
  auto mat = m_Materials.at(0);
  m_PipelineLayout = mat->m_PipelineInfo.pipelineLayout;

  // Create VkObject for each Object in the Scene
  for (const auto &object : m_Scene->GetObjects()) {
    auto material = m_Materials[object->m_Material->m_MaterialID];
    m_VkObjects.push_back(std::make_shared<VkObject>(object, material));
  }

  PLOG_DEBUG << "Done creating VkScene for Scene ID: " << m_Scene->m_SceneID;
}

VkScene::~VkScene() {
  vkDestroyDescriptorPool(vke::Device, m_DescriptorPool, nullptr);
  vkDestroyPipelineLayout(vke::Device, m_PipelineLayout, nullptr);
}

void VkScene::Draw(VkCommandBuffer commandBuffer) {
  PushConstants constants;
  auto camera = m_Scene->GetCamera();
  constants.view = camera->GetViewMatrix();
  constants.proj = camera->GetProjectionMatrix();
  vkCmdPushConstants(commandBuffer, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(PushConstants), &constants);

  for (const auto &vkObject : m_VkObjects) {
    vkObject->Draw(commandBuffer);
  }
}