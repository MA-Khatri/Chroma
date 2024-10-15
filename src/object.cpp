#include "object.h"

#include "vulkan/vulkan_utils.h"


Object::Object(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material)
    : m_Mesh(mesh), m_Material(material)
{
    // TODO?
}

Object::~Object()
{
    vkDestroyBuffer(vk::Device, m_IndexBuffer, nullptr);
    vkFreeMemory(vk::Device, m_IndexBufferMemory, nullptr);

    vkDestroyBuffer(vk::Device, m_VertexBuffer, nullptr);
    vkFreeMemory(vk::Device, m_VertexBufferMemory, nullptr);

    vkDestroyBuffer(vk::Device, m_UniformBuffer, nullptr);
    vkFreeMemory(vk::Device, m_UniformBufferMemory, nullptr);
}


void Object::VkSetup()
{
    /* Note: This MUST be called after Material::VkSetup()! */

    /* Clear descriptor writes */
    m_DescriptorWrites.resize(0);

    vk::CreateVertexBuffer(m_Mesh->vertices, m_VertexBuffer, m_VertexBufferMemory);
    vk::CreateIndexBuffer(m_Mesh->indices, m_IndexBuffer, m_IndexBufferMemory);

    /* Create buffer and device memory for the uniforms of this object */
    vk::CreateUniformBuffer(sizeof(UniformBufferObject), m_UniformBuffer, m_UniformBufferMemory, m_UniformBufferMapped);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_UniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    VkWriteDescriptorSet uboWrite{};
    uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uboWrite.dstSet = m_Material->GetDescriptorSet();
    uboWrite.dstBinding = 0;
    uboWrite.dstArrayElement = 0;
    uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboWrite.descriptorCount = 1;
    uboWrite.pBufferInfo = &bufferInfo;
    uboWrite.pImageInfo = nullptr; /* optional */
    uboWrite.pTexelBufferView = nullptr; /* optional */
    m_DescriptorWrites.push_back(uboWrite);

    vkUpdateDescriptorSets(vk::Device, static_cast<uint32_t>(m_DescriptorWrites.size()), m_DescriptorWrites.data(), 0, nullptr);
    VkUpdateUniformBuffer(); /* Need to call this just to make sure it gets set. */
}


void Object::VkDraw(VkCommandBuffer& commandBuffer)
{
#ifndef _DEBUG
    vkCmdSetDepthTestEnable(commandBuffer, m_Material->m_DepthTest); /* WARNING: For some reason, this raises an error *only* in Debug mode... */
#endif
    vkCmdSetLineWidth(commandBuffer, m_Mesh->lineWidth);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Material->m_PipelineLayout, 0, 1, &m_Material->m_DescriptorSet, 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Material->m_Pipeline);

    VkBuffer vertexBuffers[] = { m_VertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, m_IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(m_Mesh->indices.size()), 1, 0, 0, 0);
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

        ubo.color = m_Material->m_Color;

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