#include "material.h"
#include "vulkan/vulkan_utils.h"


Material::Material(TexturePaths texturePaths, int vkPipelineType, int rtMaterialType)
    : m_TexturePaths(texturePaths), m_PipelineType(vkPipelineType), m_RTMaterialType(rtMaterialType)
{
    LoadTextures();

}


Material::~Material()
{
    /*
     * Note: local pixel data of textures should be deleted automatically upon
     * destruction of the material along with the destruction of each `Texture`
     */

    vkDestroyImageView(vk::Device, m_DiffuseTextureImageView, nullptr);
    vkDestroyImage(vk::Device, m_DiffuseTextureImage, nullptr);
    vkFreeMemory(vk::Device, m_DiffuseTextureImageMemory, nullptr);
    vkDestroySampler(vk::Device, m_DiffuseTextureSampler, nullptr);

    vkDestroyImageView(vk::Device, m_SpecularTextureImageView, nullptr);
    vkDestroyImage(vk::Device, m_SpecularTextureImage, nullptr);
    vkFreeMemory(vk::Device, m_SpecularTextureImageMemory, nullptr);
    vkDestroySampler(vk::Device, m_SpecularTextureSampler, nullptr);

    vkDestroyImageView(vk::Device, m_NormalTextureImageView, nullptr);
    vkDestroyImage(vk::Device, m_NormalTextureImage, nullptr);
    vkFreeMemory(vk::Device, m_NormalTextureImageMemory, nullptr);
    vkDestroySampler(vk::Device, m_NormalTextureSampler, nullptr);
}


void Material::LoadTextures()
{
    /* Load all textures... */
    if (!m_TexturePaths.diffuse.empty())
    {
        m_DiffuseTexture.filePath = m_TexturePaths.diffuse;
        m_DiffuseTexture.LoadTexture();
    }
    if (!m_TexturePaths.specular.empty())
    {
        m_SpecularTexture.filePath = m_TexturePaths.specular;
        m_SpecularTexture.LoadTexture();
    }
    if (!m_TexturePaths.normal.empty())
    {
        m_NormalTexture.filePath = m_TexturePaths.normal;
        m_NormalTexture.LoadTexture();
    }
}


void Material::VkSetup(const PipelineInfo& pipelineInfo)
{
    m_DescriptorSetLayout = pipelineInfo.descriptorSetLayout;
    m_DescriptorPool = pipelineInfo.descriptorPool;
    m_PipelineLayout = pipelineInfo.pipelineLayout;
    m_Pipeline = pipelineInfo.pipeline;

    /* Create descriptor set for this object */
    vk::CreateDescriptorSet(m_DescriptorSetLayout, m_DescriptorPool, m_DescriptorSet);

    /* === Textures === */
    VkDescriptorImageInfo diffImageInfo{};
    VkDescriptorImageInfo specImageInfo{};
    VkDescriptorImageInfo normImageInfo{};

    if (!m_TexturePaths.diffuse.empty())
    {
        vk::CreateTextureImage(m_DiffuseTexture, m_DiffuseMipLevels, m_DiffuseTextureImage, m_DiffuseTextureImageMemory);
        vk::CreateTextureImageView(m_DiffuseMipLevels, m_DiffuseTextureImage, m_DiffuseTextureImageView);
        vk::CreateTextureSampler(m_DiffuseMipLevels, m_DiffuseTextureSampler);

        diffImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        diffImageInfo.imageView = m_DiffuseTextureImageView;
        diffImageInfo.sampler = m_DiffuseTextureSampler;

        VkWriteDescriptorSet samplerWrite{};
        samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        samplerWrite.dstSet = m_DescriptorSet;
        samplerWrite.dstBinding = 1;
        samplerWrite.dstArrayElement = 0;
        samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerWrite.descriptorCount = 1;
        samplerWrite.pImageInfo = &diffImageInfo;
        m_DescriptorWrites.push_back(samplerWrite);
    }
    if (!m_TexturePaths.specular.empty())
    {
        vk::CreateTextureImage(m_SpecularTexture, m_SpecularMipLevels, m_SpecularTextureImage, m_SpecularTextureImageMemory);
        vk::CreateTextureImageView(m_SpecularMipLevels, m_SpecularTextureImage, m_SpecularTextureImageView);
        vk::CreateTextureSampler(m_SpecularMipLevels, m_SpecularTextureSampler);

        specImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        specImageInfo.imageView = m_SpecularTextureImageView;
        specImageInfo.sampler = m_SpecularTextureSampler;

        VkWriteDescriptorSet samplerWrite{};
        samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        samplerWrite.dstSet = m_DescriptorSet;
        samplerWrite.dstBinding = 2;
        samplerWrite.dstArrayElement = 0;
        samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerWrite.descriptorCount = 1;
        samplerWrite.pImageInfo = &specImageInfo;
        m_DescriptorWrites.push_back(samplerWrite);
    }
    if (!m_TexturePaths.normal.empty())
    {
        vk::CreateTextureImage(m_NormalTexture, m_NormalMipLevels, m_NormalTextureImage, m_NormalTextureImageMemory);
        vk::CreateTextureImageView(m_NormalMipLevels, m_NormalTextureImage, m_NormalTextureImageView);
        vk::CreateTextureSampler(m_NormalMipLevels, m_NormalTextureSampler);

        normImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normImageInfo.imageView = m_NormalTextureImageView;
        normImageInfo.sampler = m_NormalTextureSampler;

        VkWriteDescriptorSet samplerWrite{};
        samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        samplerWrite.dstSet = m_DescriptorSet;
        samplerWrite.dstBinding = 3;
        samplerWrite.dstArrayElement = 0;
        samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerWrite.descriptorCount = 1;
        samplerWrite.pImageInfo = &normImageInfo;
        m_DescriptorWrites.push_back(samplerWrite);
    }
}