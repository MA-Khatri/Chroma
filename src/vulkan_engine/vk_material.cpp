#include "vk_material.hpp"
#include "vulkan_engine.hpp"
#include <vulkan/vulkan_core.h>

VkDescriptorSetLayoutBinding CreateDSLFragmentBinding(unsigned int binding) {
  VkDescriptorSetLayoutBinding layoutBinding{};
  layoutBinding.binding = binding;
  layoutBinding.descriptorCount = 1;
  layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  layoutBinding.pImmutableSamplers = nullptr;
  layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  return layoutBinding;
}

VkMaterial::VkMaterial(std::shared_ptr<Material> material, VkDescriptorPool descriptorPool,
                       VkDescriptorSetLayout sceneDescriptorSetLayout,
                       VkSampleCountFlagBits msaaCount, VkRenderPass renderPass,
                       VkRenderPass pickRenderPass) {
  // Clear the descriptor writes
  m_DescriptorWrites.resize(0);

  // Descriptor set layout creation: uniforms, textures/samplers
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

  // We'll have 1 ubo to pass in mesh data like its model & normal matrices
  VkDescriptorSetLayoutBinding objectUboBinding{};
  objectUboBinding.binding = 0;
  objectUboBinding.descriptorCount = 1;
  objectUboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  objectUboBinding.pImmutableSamplers = nullptr;
  objectUboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  layoutBindings.push_back(objectUboBinding);

  // Create graphics pipeline
  std::vector<vke::ShaderInfo> shaders;
  std::vector<vke::ShaderInfo> pickShaders;
  VkPrimitiveTopology topology;

  if (material->m_Type < MaterialType::PointFlat) { // Surfaces
    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    shaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Surface.vert.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Surface.vert.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/DepthOnly.frag.spv"});
  } else if (material->m_Type < MaterialType::Lines) { // Points
    topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    shaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Point.vert.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Point.vert.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/DepthOnlyPoint.frag.spv"});
  } else { // Lines
    topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  }

  bool hasTextures = material->HasTextures();
  bool hasPick = true;

  switch (material->m_Type) {
  case MaterialType::SurfacePerVertexFlat:
    shaders.push_back(
        {VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/SurfacePerVertexFlat.frag.spv"});
    break;

  case MaterialType::SurfacePerVertexShaded:
    shaders.push_back(
        {VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/SurfacePerVertexShaded.frag.spv"});
    break;

  case MaterialType::SurfacePerVertexNormal:
    shaders.push_back(
        {VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/SurfacePerVertexNormal.frag.spv"});
    break;

  case MaterialType::SurfaceAlbedoTextureFlat:
    shaders.push_back(
        {VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/SurfaceAlbedoTextureFlat.frag.spv"});
    layoutBindings.push_back(CreateDSLFragmentBinding(1));
    break;

  case MaterialType::SurfaceAlbedoTextureShaded:
    shaders.push_back(
        {VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/SurfaceAlbedoTextureShaded.frag.spv"});
    layoutBindings.push_back(CreateDSLFragmentBinding(1));
    break;

  case MaterialType::PointFlat:
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/PointFlat.frag.spv"});
    break;

  case MaterialType::PointShaded:
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/PointShaded.frag.spv"});
    break;

  case MaterialType::PointNormal:
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/PointNormal.frag.spv"});
    break;

  case MaterialType::Lines:
    shaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Default.vert.spv"});
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/Default.frag.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Default.vert.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/DepthOnly.frag.spv"});
    break;

  case MaterialType::GroundGrid:
    shaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/GroundGrid.vert.spv"});
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/GroundGrid.frag.spv"});
    hasPick = false;
    break;

  case MaterialType::OrientationGizmo:
    shaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/OrientationGizmo.vert.spv"});
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/Default.frag.spv"});
    hasPick = false;
    break;

  case MaterialType::Reticle:
    shaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Reticle.vert.spv"});
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/Default.frag.spv"});
    hasPick = false;
    break;

  default:
    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    shaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Default.vert.spv"});
    shaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/Default.frag.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vulkan_shaders/Default.vert.spv"});
    pickShaders.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "vulkan_shaders/DepthOnly.frag.spv"});

    PLOG_WARNING << "Material type not recognized, using default shader: "
                 << static_cast<int>(material->m_Type);
    break;
  }

  // Create descriptor set layout
  vke::CreateDescriptorSetLayout(layoutBindings, m_PipelineInfo.objectDescriptorSetLayout);

  std::vector<VkDescriptorSetLayout> DSLs = {
      sceneDescriptorSetLayout,                // set 0
      m_PipelineInfo.objectDescriptorSetLayout // set 1
  };
  m_PipelineInfo.pipeline = vke::CreateGraphicsPipeline(shaders, msaaCount, topology, renderPass,
                                                        DSLs, m_PipelineInfo.pipelineLayout);

  if (pickRenderPass != VK_NULL_HANDLE && hasPick) {
    m_PipelineInfo.pickPipeline =
        vke::CreateGraphicsPipeline(pickShaders, VK_SAMPLE_COUNT_1_BIT, topology, pickRenderPass,
                                    DSLs, m_PipelineInfo.pipelineLayout);
  }

  // Store texture writes for later binding on the per-object descriptor set.
  m_PipelineInfo.descriptorPool = descriptorPool;

  // === Textures ===
  if (hasTextures) {
    if (!material->m_AlbedoTexture.empty()) {
      VkDescriptorImageInfo &diffImageInfo = m_DescriptorImageInfos[m_DescriptorImageInfoCount++];
      vke::CreateTextureImage(material->m_AlbedoTexture, m_AlbedoMipLevels, m_AlbedoTextureImage,
                              m_AlbedoTextureImageMemory);
      vke::CreateTextureImageView(m_AlbedoMipLevels, m_AlbedoTextureImage,
                                  m_AlbedoTextureImageView);
      vke::CreateTextureSampler(m_AlbedoMipLevels, m_AlbedoTextureSampler);

      diffImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      diffImageInfo.imageView = m_AlbedoTextureImageView;
      diffImageInfo.sampler = m_AlbedoTextureSampler;

      VkWriteDescriptorSet samplerWrite{};
      samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      samplerWrite.dstSet = VK_NULL_HANDLE;
      samplerWrite.dstBinding = 1;
      samplerWrite.dstArrayElement = 0;
      samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      samplerWrite.descriptorCount = 1;
      samplerWrite.pImageInfo = &diffImageInfo;
      m_DescriptorWrites.push_back(samplerWrite);
    }

    // if (!material->m_RoughnessTexture.empty()) {
    //   VkDescriptorImageInfo &specImageInfo =
    //   m_DescriptorImageInfos[m_DescriptorImageInfoCount++];
    //   vke::CreateTextureImage(material->m_RoughnessTexture, m_SpecularMipLevels,
    //                           m_SpecularTextureImage, m_SpecularTextureImageMemory);
    //   vke::CreateTextureImageView(m_SpecularMipLevels, m_SpecularTextureImage,
    //                               m_SpecularTextureImageView);
    //   vke::CreateTextureSampler(m_SpecularMipLevels, m_SpecularTextureSampler);

    //   specImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    //   specImageInfo.imageView = m_SpecularTextureImageView;
    //   specImageInfo.sampler = m_SpecularTextureSampler;

    //   VkWriteDescriptorSet samplerWrite{};
    //   samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    //   samplerWrite.dstSet = VK_NULL_HANDLE;
    //   samplerWrite.dstBinding = 2;
    //   samplerWrite.dstArrayElement = 0;
    //   samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    //   samplerWrite.descriptorCount = 1;
    //   samplerWrite.pImageInfo = &specImageInfo;
    //   m_DescriptorWrites.push_back(samplerWrite);
    // }

    // if (!material->m_NormalTexture.empty()) {
    //   VkDescriptorImageInfo &normImageInfo =
    //   m_DescriptorImageInfos[m_DescriptorImageInfoCount++];
    //   vke::CreateTextureImage(material->m_NormalTexture, m_NormalMipLevels, m_NormalTextureImage,
    //                           m_NormalTextureImageMemory);
    //   vke::CreateTextureImageView(m_NormalMipLevels, m_NormalTextureImage,
    //                               m_NormalTextureImageView);
    //   vke::CreateTextureSampler(m_NormalMipLevels, m_NormalTextureSampler);

    //   normImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    //   normImageInfo.imageView = m_NormalTextureImageView;
    //   normImageInfo.sampler = m_NormalTextureSampler;

    //   VkWriteDescriptorSet samplerWrite{};
    //   samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    //   samplerWrite.dstSet = VK_NULL_HANDLE;
    //   samplerWrite.dstBinding = 3;
    //   samplerWrite.dstArrayElement = 0;
    //   samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    //   samplerWrite.descriptorCount = 1;
    //   samplerWrite.pImageInfo = &normImageInfo;
    //   m_DescriptorWrites.push_back(samplerWrite);
    // }
  }
}

VkMaterial::~VkMaterial() {
  // Cleanup textures
  vkDestroyImageView(vke::Device, m_AlbedoTextureImageView, nullptr);
  vkDestroyImage(vke::Device, m_AlbedoTextureImage, nullptr);
  vkFreeMemory(vke::Device, m_AlbedoTextureImageMemory, nullptr);
  vkDestroySampler(vke::Device, m_AlbedoTextureSampler, nullptr);

  // vkDestroyImageView(vke::Device, m_RoughnessTextureImageView, nullptr);
  // vkDestroyImage(vke::Device, m_RoughnessTextureImage, nullptr);
  // vkFreeMemory(vke::Device, m_RoughnessTextureImageMemory, nullptr);
  // vkDestroySampler(vke::Device, m_RoughnessTextureSampler, nullptr);

  // vkDestroyImageView(vke::Device, m_NormalTextureImageView, nullptr);
  // vkDestroyImage(vke::Device, m_NormalTextureImage, nullptr);
  // vkFreeMemory(vke::Device, m_NormalTextureImageMemory, nullptr);
  // vkDestroySampler(vke::Device, m_NormalTextureSampler, nullptr);
}