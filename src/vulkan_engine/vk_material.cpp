#include "vk_material.hpp"
#include "vulkan_engine.hpp"
#include <vulkan/vulkan_core.h>

VkMaterial::VkMaterial(std::shared_ptr<Material> material, VkDescriptorPool descriptorPool,
                       ImVec2 viewportSize, VkSampleCountFlagBits msaaCount,
                       VkRenderPass renderPass) {
  // Clear the descriptor writes
  m_DescriptorWrites.resize(0);

  // TODO: Create pipeline info based on material type
  // For now, we create the same pipeline info for all materials

  // Descriptor set layout creation: uniforms, textures/samplers
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

  // We'll have 1 ubo to pass in mesh data like its model & normal matrices
  VkDescriptorSetLayoutBinding uboLayoutBinding{};
  uboLayoutBinding.binding = 0;
  uboLayoutBinding.descriptorCount = 1;
  uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uboLayoutBinding.pImmutableSamplers = nullptr;
  uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  layoutBindings.push_back(uboLayoutBinding);

  // We'll have 3 samplers for diffuse, specular, and normal textures
  VkDescriptorSetLayoutBinding diffuseSamplerLayoutBinding{};
  diffuseSamplerLayoutBinding.binding = 1;
  diffuseSamplerLayoutBinding.descriptorCount = 1;
  diffuseSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  diffuseSamplerLayoutBinding.pImmutableSamplers = nullptr;
  diffuseSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  layoutBindings.push_back(diffuseSamplerLayoutBinding);

  VkDescriptorSetLayoutBinding specularSamplerLayoutBinding{};
  specularSamplerLayoutBinding.binding = 2;
  specularSamplerLayoutBinding.descriptorCount = 1;
  specularSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  specularSamplerLayoutBinding.pImmutableSamplers = nullptr;
  specularSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  layoutBindings.push_back(specularSamplerLayoutBinding);

  VkDescriptorSetLayoutBinding normalSamplerLayoutBinding{};
  normalSamplerLayoutBinding.binding = 3;
  normalSamplerLayoutBinding.descriptorCount = 1;
  normalSamplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  normalSamplerLayoutBinding.pImmutableSamplers = nullptr;
  normalSamplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  layoutBindings.push_back(normalSamplerLayoutBinding);

  // Create descriptor set layout
  vke::CreateDescriptorSetLayout(layoutBindings, m_PipelineInfo.descriptorSetLayout);

  // Create graphics pipeline
  std::vector<std::string> shaderFiles;
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  switch (material->m_Type) {
  case MaterialType::Lambertian:
  case MaterialType::Conductor:
  case MaterialType::Dielectric:
  case MaterialType::Principled:
  case MaterialType::Emissive:
    shaderFiles = {
        "vulkan_shaders/Solid.vert.spv",
        "vulkan_shaders/Solid.frag.spv",
    };
    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    break;

  case MaterialType::Point:
    // TODO: write point shaders
    shaderFiles = {
        "vulkan_shaders/Solid.vert.spv",
        "vulkan_shaders/Solid.frag.spv",
    };
    topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    break;

  case MaterialType::Lines:
    // TODO: write line shaders
    shaderFiles = {
        "vulkan_shaders/Solid.vert.spv",
        "vulkan_shaders/Solid.frag.spv",
    };
    topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    break;

  case MaterialType::GroundGrid:
    shaderFiles = {
        "vulkan_shaders/GroundGrid.vert.spv",
        "vulkan_shaders/GroundGrid.frag.spv",
    };
    topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    break;

  default:
    shaderFiles = {
        "vulkan_shaders/Solid.vert.spv",
        "vulkan_shaders/Solid.frag.spv",
    };
    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    break;
  }

  m_PipelineInfo.pipeline = vke::CreateGraphicsPipeline(
      shaderFiles, viewportSize, msaaCount, topology, renderPass,
      m_PipelineInfo.descriptorSetLayout, m_PipelineInfo.pipelineLayout);

  // Store texture writes for later binding on the per-object descriptor set.
  m_PipelineInfo.descriptorPool = descriptorPool;

  // === Textures ===
  if (!material->m_AlbedoTexture.m_Pixels.empty()) {
    VkDescriptorImageInfo &diffImageInfo = m_DescriptorImageInfos[m_DescriptorImageInfoCount++];
    vke::CreateTextureImage(material->m_AlbedoTexture, m_DiffuseMipLevels, m_DiffuseTextureImage,
                            m_DiffuseTextureImageMemory);
    vke::CreateTextureImageView(m_DiffuseMipLevels, m_DiffuseTextureImage,
                                m_DiffuseTextureImageView);
    vke::CreateTextureSampler(m_DiffuseMipLevels, m_DiffuseTextureSampler);

    diffImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    diffImageInfo.imageView = m_DiffuseTextureImageView;
    diffImageInfo.sampler = m_DiffuseTextureSampler;

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
  if (!material->m_RoughnessTexture.m_Pixels.empty()) {
    VkDescriptorImageInfo &specImageInfo = m_DescriptorImageInfos[m_DescriptorImageInfoCount++];
    vke::CreateTextureImage(material->m_RoughnessTexture, m_SpecularMipLevels,
                            m_SpecularTextureImage, m_SpecularTextureImageMemory);
    vke::CreateTextureImageView(m_SpecularMipLevels, m_SpecularTextureImage,
                                m_SpecularTextureImageView);
    vke::CreateTextureSampler(m_SpecularMipLevels, m_SpecularTextureSampler);

    specImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    specImageInfo.imageView = m_SpecularTextureImageView;
    specImageInfo.sampler = m_SpecularTextureSampler;

    VkWriteDescriptorSet samplerWrite{};
    samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    samplerWrite.dstSet = VK_NULL_HANDLE;
    samplerWrite.dstBinding = 2;
    samplerWrite.dstArrayElement = 0;
    samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerWrite.descriptorCount = 1;
    samplerWrite.pImageInfo = &specImageInfo;
    m_DescriptorWrites.push_back(samplerWrite);
  }
  if (!material->m_NormalTexture.m_Pixels.empty()) {
    VkDescriptorImageInfo &normImageInfo = m_DescriptorImageInfos[m_DescriptorImageInfoCount++];
    vke::CreateTextureImage(material->m_NormalTexture, m_NormalMipLevels, m_NormalTextureImage,
                            m_NormalTextureImageMemory);
    vke::CreateTextureImageView(m_NormalMipLevels, m_NormalTextureImage, m_NormalTextureImageView);
    vke::CreateTextureSampler(m_NormalMipLevels, m_NormalTextureSampler);

    normImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    normImageInfo.imageView = m_NormalTextureImageView;
    normImageInfo.sampler = m_NormalTextureSampler;

    VkWriteDescriptorSet samplerWrite{};
    samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    samplerWrite.dstSet = VK_NULL_HANDLE;
    samplerWrite.dstBinding = 3;
    samplerWrite.dstArrayElement = 0;
    samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerWrite.descriptorCount = 1;
    samplerWrite.pImageInfo = &normImageInfo;
    m_DescriptorWrites.push_back(samplerWrite);
  }
  // TODO: add other textures as needed
}

VkMaterial::~VkMaterial() {
  // Cleanup textures
  vkDestroyImageView(vke::Device, m_DiffuseTextureImageView, nullptr);
  vkDestroyImage(vke::Device, m_DiffuseTextureImage, nullptr);
  vkFreeMemory(vke::Device, m_DiffuseTextureImageMemory, nullptr);
  vkDestroySampler(vke::Device, m_DiffuseTextureSampler, nullptr);

  vkDestroyImageView(vke::Device, m_SpecularTextureImageView, nullptr);
  vkDestroyImage(vke::Device, m_SpecularTextureImage, nullptr);
  vkFreeMemory(vke::Device, m_SpecularTextureImageMemory, nullptr);
  vkDestroySampler(vke::Device, m_SpecularTextureSampler, nullptr);

  vkDestroyImageView(vke::Device, m_NormalTextureImageView, nullptr);
  vkDestroyImage(vke::Device, m_NormalTextureImage, nullptr);
  vkFreeMemory(vke::Device, m_NormalTextureImageMemory, nullptr);
  vkDestroySampler(vke::Device, m_NormalTextureSampler, nullptr);
}