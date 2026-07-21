#include "vk_scene.hpp"
#include "vulkan_utils.hpp"
#include <map>
#include <set>
#include <vulkan/vulkan_core.h>

VkScene::VkScene(std::shared_ptr<Scene> scene, VkSampleCountFlagBits msaaCount,
                 VkRenderPass renderPass)
    : m_Scene(scene) {
  PLOG_DEBUG << "Creating VkScene for Scene ID: " << m_Scene->m_SceneID;

  // Create descriptor pool
  std::set<int> materialIDs;
  for (const auto &object : m_Scene->GetObjects()) {
    materialIDs.insert(object->m_Material->m_MaterialID);
  }
  vke::CreateDescriptorPool(materialIDs.size(), m_Scene->GetObjects().size(), m_DescriptorPool);

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

  VkUploadUniformBuffer(ImVec2(1, 1));

  // Initialize unique materials
  for (const auto &object : m_Scene->GetObjects()) {
    int matID = object->m_Material->m_MaterialID;
    if (m_Materials.find(matID) == m_Materials.end()) {
      m_Materials.insert(
          {matID, std::make_shared<VkMaterial>(object->m_Material, m_DescriptorPool,
                                               m_DescriptorSetLayout, msaaCount, renderPass)});
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

void VkScene::Draw(VkCommandBuffer commandBuffer, ImVec2 viewportSize) {
  // Upload and bind scene UBO descriptor set
  VkUploadUniformBuffer(viewportSize);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1,
                          &m_DescriptorSet, 0, nullptr);

  for (const auto &vkObject : m_VkObjects) {
    vkObject->Draw(commandBuffer);
  }
}

void VkScene::VkUploadUniformBuffer(ImVec2 viewportSize) {
  if (m_UniformBufferMapped) {
    auto camera = m_Scene->GetCamera();

    SceneUBO ubo{};
    ubo.viewMatrix = camera->GetViewMatrix();
    ubo.projectionMatrix = camera->GetProjectionMatrix();
    ubo.viewProjectionMatrix = camera->GetViewProjectionMatrix();
    ubo.cameraPositionAndViewportHeight = glm::vec4(camera->GetPosition(), viewportSize.y);

    memcpy(m_UniformBufferMapped, &ubo, sizeof(ubo));
  } else {
    PLOG_WARNING << "m_UniformBufferMapped is nullptr!" << std::endl;
  }
}
