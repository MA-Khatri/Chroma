#include "object.h"

#include "stb_image.h"
#include "vulkan/vulkan_utils.h"


Object::Object(Mesh mesh, TexturePaths texturePaths, int pipelineType)
    : m_Mesh(mesh), m_TexturePaths(texturePaths), m_PipelineType(pipelineType)
{
    /* Load all textures... */
    if (!m_TexturePaths.diffuse.empty())
    {
        m_DiffuseTexture.filePath = m_TexturePaths.diffuse;
        LoadTexture(m_DiffuseTexture);
    }
    if (!m_TexturePaths.specular.empty())
    {
        m_SpecularTexture.filePath = m_TexturePaths.specular;
        LoadTexture(m_SpecularTexture);
    }
    if (!m_TexturePaths.normal.empty())
    {
        m_NormalTexture.filePath = m_TexturePaths.normal;
        LoadTexture(m_NormalTexture);
    }
}


Object::~Object()
{
    /* 
     * Note: local pixel data of textures should be deleted automatically upon
     * destruction of the object along with the destruction of each `Texture`
     */

    vkDestroyBuffer(vk::Device, m_IndexBuffer, nullptr);
    vkFreeMemory(vk::Device, m_IndexBufferMemory, nullptr);

    vkDestroyBuffer(vk::Device, m_VertexBuffer, nullptr);
    vkFreeMemory(vk::Device, m_VertexBufferMemory, nullptr);

    vkDestroyBuffer(vk::Device, m_UniformBuffer, nullptr);
    vkFreeMemory(vk::Device, m_UniformBufferMemory, nullptr);

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


void Object::LoadTexture(Texture& tex)
{
    int texWidth, texHeight, texChannels;
    uint8_t* pixels = stbi_load(tex.filePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); /* load image with alpha channel even if it doesn't have one */
    if (!pixels)
    {
        std::cerr << "LoadTexture(): Error! Failed to load image " << tex.filePath << " ! " << std::endl;
        exit(-1);
    }

#ifdef _DEBUG
    if (texChannels != 4) std::cout << "Warning: loaded texture only has " << texChannels << " channels. Force loaded with 4 channels!" << std::endl;
#endif
    tex.resolution = glm::ivec3(texWidth, texHeight, 4);

    /* Copy pixels to local std::vector and free originally read data */
    tex.pixels = std::vector<uint8_t>(pixels, pixels + (tex.resolution.x * tex.resolution.y * tex.resolution.z));
    stbi_image_free(pixels);
}


void Object::VkSetup(const PipelineInfo& pipelineInfo)
{
    vk::CreateVertexBuffer(m_Mesh.vertices, m_VertexBuffer, m_VertexBufferMemory);
    vk::CreateIndexBuffer(m_Mesh.indices, m_IndexBuffer, m_IndexBufferMemory);

    m_DescriptorSetLayout = pipelineInfo.descriptorSetLayout;
    m_DescriptorPool = pipelineInfo.descriptorPool;
    m_PipelineLayout = pipelineInfo.pipelineLayout;
    m_Pipeline = pipelineInfo.pipeline;

    /* Create descriptor set for this object */
    vk::CreateDescriptorSet(m_DescriptorSetLayout, m_DescriptorPool, m_DescriptorSet);
    std::vector<VkWriteDescriptorSet> descriptorWrites;

    /* Create buffer and device memory for the uniforms of this object */
    vk::CreateUniformBuffer(sizeof(UniformBufferObject), m_UniformBuffer, m_UniformBufferMemory, m_UniformBufferMapped);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_UniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    VkWriteDescriptorSet uboWrite{};
    uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uboWrite.dstSet = m_DescriptorSet;
    uboWrite.dstBinding = 0;
    uboWrite.dstArrayElement = 0;
    uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboWrite.descriptorCount = 1;
    uboWrite.pBufferInfo = &bufferInfo;
    uboWrite.pImageInfo = nullptr; /* optional */
    uboWrite.pTexelBufferView = nullptr; /* optional */
    descriptorWrites.push_back(uboWrite);

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
        descriptorWrites.push_back(samplerWrite);
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
        descriptorWrites.push_back(samplerWrite);
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
        descriptorWrites.push_back(samplerWrite);
    }

    vkUpdateDescriptorSets(vk::Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    VkUpdateUniformBuffer(); /* Need to call this just to make sure it gets set. */
}


void Object::VkDraw(VkCommandBuffer& commandBuffer)
{
#ifndef _DEBUG
    vkCmdSetDepthTestEnable(commandBuffer, m_DepthTest); /* WARNING: For some reason, this raises an error *only* in Debug mode... */
#endif
    vkCmdSetLineWidth(commandBuffer, m_Mesh.lineWidth);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

    VkBuffer vertexBuffers[] = { m_VertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_Mesh.indices.size()), 1, 0, 0, 0);
}


void Object::VkUpdateUniformBuffer()
{
    if (m_UniformBufferMapped)
    {
        UniformBufferObject ubo{};
        ubo.modelMatrix = m_ModelMatrix;
        ubo.normalMatrix = glm::mat4(m_ModelNormalMatrix);

        //std::cout << ubo.normalmatrix[0][0] << " " << ubo.normalmatrix[1][0] << " " << ubo.normalmatrix[2][0] << std::endl;
        //std::cout << ubo.normalmatrix[0][1] << " " << ubo.normalmatrix[1][1] << " " << ubo.normalmatrix[2][1] << std::endl;
        //std::cout << ubo.normalmatrix[0][2] << " " << ubo.normalmatrix[1][2] << " " << ubo.normalmatrix[2][2] << std::endl;
        //std::cout << std::endl;

        ubo.color = m_Color;

        memcpy(m_UniformBufferMapped, &ubo, sizeof(ubo));
    }
#ifdef _DEBUG
    else
    {
        std::cerr << "VkUpdateUniformBuffer(): Warning, m_UniformBufferMapped is nullptr!" << std::endl;
    }
#endif
}


/* === Transformations === */

void Object::UpdateModelNormalMatrix()
{
    /* Take the upper left 3x3 of the model matrix then inverse transpose to get 3x3 model normal */
    m_ModelNormalMatrix = glm::transpose(glm::inverse(glm::mat3(m_ModelMatrix)));
}


void Object::SetModelMatrix(glm::mat4 matrix)
{
    m_ModelMatrix = matrix;
    UpdateModelNormalMatrix();
    VkUpdateUniformBuffer();
}


void Object::UpdateModelMatrix(glm::mat4 matrix)
{
    m_ModelMatrix *= matrix;
    UpdateModelNormalMatrix();
    VkUpdateUniformBuffer();
}


void Object::Translate(glm::vec3 translate)
{
    m_ModelMatrix *= glm::translate(translate);
    VkUpdateUniformBuffer();
}


void Object::Translate(float x, float y, float z)
{
    Translate(glm::vec3(x, y, z));
}


void Object::Rotate(glm::vec3 axis, float deg)
{
    m_ModelMatrix *= glm::rotate(glm::radians(deg), axis);
    UpdateModelNormalMatrix();
    VkUpdateUniformBuffer();
}


void Object::Scale(glm::vec3 scale, bool updateNormal /* = true */)
{
    m_ModelMatrix *= glm::scale(scale);
    if (updateNormal)
    {
        UpdateModelNormalMatrix();
    }
    VkUpdateUniformBuffer();
}


void Object::Scale(float scale)
{
    Scale(glm::vec3(scale, scale, scale), false);
}


void Object::Scale(float x, float y, float z)
{
    Scale(glm::vec3(x, y, z), true);
}