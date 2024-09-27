#include "object.h"

#include "vulkan_utils.h"


Object::Object(Mesh mesh, VkDescriptorSetLayout& descriptorSetLayout, VkDescriptorPool& descriptorPool, VkPipelineLayout& pipelineLayout, VkPipeline& pipeline)
{
    m_Mesh = mesh;
    VK::CreateVertexBuffer(m_Mesh.vertices, m_VertexBuffer, m_VertexBufferMemory);
    VK::CreateIndexBuffer(m_Mesh.indices, m_IndexBuffer, m_IndexBufferMemory);

    m_DescriptorSetLayout = descriptorSetLayout;
    m_DescriptorPool = descriptorPool;
    m_PipelineLayout = pipelineLayout;
    m_Pipeline = pipeline;

    /* Textures */
    //VK::CreateTextureImage("res/textures/texture.jpg", m_TextureImage, m_TextureImageMemory);
    VK::CreateTextureImage("res/textures/viking_room.png", m_TextureImage, m_TextureImageMemory);
    VK::CreateTextureImageView(m_TextureImage, m_TextureImageView);
    VK::CreateTextureSampler(m_TextureSampler);

    /* Create buffers and device memory for the uniforms of this object */
    VK::CreateUniformBuffers(sizeof(UniformBufferObject), m_UniformBuffers, m_UniformBuffersMemory, m_UniformBuffersMapped);

    /* Create descriptor sets for this object */
    VK::CreateDescriptorSets(m_DescriptorSetLayout, m_DescriptorPool, m_DescriptorSets);
    for (size_t i = 0; i < VK::ImageCount; i++)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_UniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_TextureImageView;
        imageInfo.sampler = m_TextureSampler;

        std::vector<VkWriteDescriptorSet> descriptorWrites;

        VkWriteDescriptorSet uboWrite{};
        uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        uboWrite.dstSet = m_DescriptorSets[i];
        uboWrite.dstBinding = 0;
        uboWrite.dstArrayElement = 0;
        uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboWrite.descriptorCount = 1;
        uboWrite.pBufferInfo = &bufferInfo;
        uboWrite.pImageInfo = nullptr; /* optional */
        uboWrite.pTexelBufferView = nullptr; /* optional */
        descriptorWrites.push_back(uboWrite);

        VkWriteDescriptorSet samplerWrite{};
        samplerWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        samplerWrite.dstSet = m_DescriptorSets[i];
        samplerWrite.dstBinding = 1;
        samplerWrite.dstArrayElement = 0;
        samplerWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerWrite.descriptorCount = 1;
        samplerWrite.pImageInfo = &imageInfo;
        descriptorWrites.push_back(samplerWrite);

        vkUpdateDescriptorSets(VK::Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}


Object::~Object()
{
    vkDestroyBuffer(VK::Device, m_IndexBuffer, nullptr);
    vkFreeMemory(VK::Device, m_IndexBufferMemory, nullptr);

    vkDestroyBuffer(VK::Device, m_VertexBuffer, nullptr);
    vkFreeMemory(VK::Device, m_VertexBufferMemory, nullptr);

    for (size_t i = 0; i < VK::ImageCount; i++)
    {
        vkDestroyBuffer(VK::Device, m_UniformBuffers[i], nullptr);
        vkFreeMemory(VK::Device, m_UniformBuffersMemory[i], nullptr);
    }

    vkDestroySampler(VK::Device, m_TextureSampler, nullptr);
    vkDestroyImageView(VK::Device, m_TextureImageView, nullptr);
    vkDestroyImage(VK::Device, m_TextureImage, nullptr);
    vkFreeMemory(VK::Device, m_TextureImageMemory, nullptr);
}


void Object::Draw(VkCommandBuffer& commandBuffer)
{
    UpdateUniformBuffer();

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSets[VK::MainWindowData.FrameIndex], 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

    VkBuffer vertexBuffers[] = { m_VertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_Mesh.indices.size()), 1, 0, 0, 0);
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


void Object::UpdateUniformBuffer()
{
    UniformBufferObject ubo{};
    ubo.modelMatrix = m_ModelMatrix;
    ubo.normalMatrix = m_ModelNormalMatrix;

    memcpy(m_UniformBuffersMapped[VK::MainWindowData.FrameIndex], &ubo, sizeof(ubo));
}