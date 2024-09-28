#include "object.h"

#include "vulkan_utils.h"


Object::Object(Mesh mesh, TexturePaths texturePaths, const PipelineInfo& pipelineInfo)
{
    m_Mesh = mesh;
    VK::CreateVertexBuffer(m_Mesh.vertices, m_VertexBuffer, m_VertexBufferMemory);
    VK::CreateIndexBuffer(m_Mesh.indices, m_IndexBuffer, m_IndexBufferMemory);

    m_DescriptorSetLayout = pipelineInfo.descriptorSetLayout;
    m_DescriptorPool = pipelineInfo.descriptorPool;
    m_PipelineLayout = pipelineInfo.pipelineLayout;
    m_Pipeline = pipelineInfo.pipeline;

    /* Create descriptor set for this object */
    VK::CreateDescriptorSet(m_DescriptorSetLayout, m_DescriptorPool, m_DescriptorSet);
    std::vector<VkWriteDescriptorSet> descriptorWrites;

    /* Create buffer and device memory for the uniforms of this object */
    VK::CreateUniformBuffer(sizeof(UniformBufferObject), m_UniformBuffer, m_UniformBufferMemory, m_UniformBufferMapped);

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
    /* We can use the same texture sampler for all our textures */
    VK::CreateTextureSampler(m_TextureSampler);

    VkDescriptorImageInfo diffImageInfo{};
    VkDescriptorImageInfo specImageInfo{};
    VkDescriptorImageInfo normImageInfo{};

    if (!texturePaths.diffuse.empty())
    {
        VK::CreateTextureImage(texturePaths.diffuse, m_DiffuseTextureImage, m_DiffuseTextureImageMemory);
        VK::CreateTextureImageView(m_DiffuseTextureImage, m_DiffuseTextureImageView);

        diffImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        diffImageInfo.imageView = m_DiffuseTextureImageView;
        diffImageInfo.sampler = m_TextureSampler;

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
    if (!texturePaths.specular.empty())
    {
        VK::CreateTextureImage(texturePaths.specular, m_SpecularTextureImage, m_SpecularTextureImageMemory);
        VK::CreateTextureImageView(m_SpecularTextureImage, m_SpecularTextureImageView);

        specImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        specImageInfo.imageView = m_SpecularTextureImageView;
        specImageInfo.sampler = m_TextureSampler;

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
    if (!texturePaths.normal.empty())
    {
        VK::CreateTextureImage(texturePaths.normal, m_NormalTextureImage, m_NormalTextureImageMemory);
        VK::CreateTextureImageView(m_NormalTextureImage, m_NormalTextureImageView);

        normImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normImageInfo.imageView = m_NormalTextureImageView;
        normImageInfo.sampler = m_TextureSampler;

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

    vkUpdateDescriptorSets(VK::Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}


Object::~Object()
{
    vkDestroyBuffer(VK::Device, m_IndexBuffer, nullptr);
    vkFreeMemory(VK::Device, m_IndexBufferMemory, nullptr);

    vkDestroyBuffer(VK::Device, m_VertexBuffer, nullptr);
    vkFreeMemory(VK::Device, m_VertexBufferMemory, nullptr);

    vkDestroyBuffer(VK::Device, m_UniformBuffer, nullptr);
    vkFreeMemory(VK::Device, m_UniformBufferMemory, nullptr);

    vkDestroySampler(VK::Device, m_TextureSampler, nullptr);

    vkDestroyImageView(VK::Device, m_DiffuseTextureImageView, nullptr);
    vkDestroyImage(VK::Device, m_DiffuseTextureImage, nullptr);
    vkFreeMemory(VK::Device, m_DiffuseTextureImageMemory, nullptr);

    vkDestroyImageView(VK::Device, m_SpecularTextureImageView, nullptr);
    vkDestroyImage(VK::Device, m_SpecularTextureImage, nullptr);
    vkFreeMemory(VK::Device, m_SpecularTextureImageMemory, nullptr);

    vkDestroyImageView(VK::Device, m_NormalTextureImageView, nullptr);
    vkDestroyImage(VK::Device, m_NormalTextureImage, nullptr);
    vkFreeMemory(VK::Device, m_NormalTextureImageMemory, nullptr);
}


void Object::Draw(VkCommandBuffer& commandBuffer)
{
    UpdateUniformBuffer();

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

    VkBuffer vertexBuffers[] = { m_VertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_Mesh.indices.size()), 1, 0, 0, 0);
}


void Object::UpdateUniformBuffer()
{
    UniformBufferObject ubo{};
    ubo.modelMatrix = m_ModelMatrix;
    ubo.normalMatrix = m_ModelNormalMatrix;

    memcpy(m_UniformBufferMapped, &ubo, sizeof(ubo));
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
}


void Object::UpdateModelMatrix(glm::mat4 matrix)
{
    m_ModelMatrix *= matrix;
    UpdateModelNormalMatrix();
}


void Object::Translate(glm::vec3 translate)
{
    m_ModelMatrix *= glm::translate(translate);
}


void Object::Translate(float x, float y, float z)
{
    Translate(glm::vec3(x, y, z));
}


void Object::Rotate(glm::vec3 axis, float deg)
{
    m_ModelMatrix *= glm::rotate(glm::radians(deg), axis);
    UpdateModelNormalMatrix();
}


void Object::Scale(glm::vec3 scale, bool updateNormal /* = true */)
{
    m_ModelMatrix *= glm::scale(scale);
    if (updateNormal)
    {
        UpdateModelNormalMatrix();
    }
}


void Object::Scale(float scale)
{
    Scale(glm::vec3(scale, scale, scale), false);
}


void Object::Scale(float x, float y, float z)
{
    Scale(glm::vec3(x, y, z), true);
}