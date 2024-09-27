#include "object.h"

#include "vulkan_utils.h"


Object::Object(Mesh mesh, VkPipeline& pipeline)
{
    m_Mesh = mesh;
    VK::CreateVertexBuffer(m_Mesh.vertices, m_VertexBuffer, m_VertexBufferMemory);
    VK::CreateIndexBuffer(m_Mesh.indices, m_IndexBuffer, m_IndexBufferMemory);
    m_Pipeline = pipeline;
}


Object::~Object()
{
    vkDestroyBuffer(VK::Device, m_IndexBuffer, nullptr);
    vkFreeMemory(VK::Device, m_IndexBufferMemory, nullptr);

    vkDestroyBuffer(VK::Device, m_VertexBuffer, nullptr);
    vkFreeMemory(VK::Device, m_VertexBufferMemory, nullptr);
}


void Object::Draw(VkCommandBuffer& commandBuffer)
{
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
    m_ModelNormalMatrix = glm::transpose(glm::inverse(m_ModelMatrix));
}


void Object::SetModelMatrix(glm::mat4x4 matrix)
{
    m_ModelMatrix = matrix;
    UpdateModelNormalMatrix();
}


void Object::UpdateModelMatrix(glm::mat4x4 matrix)
{
    m_ModelMatrix *= matrix;
    UpdateModelNormalMatrix();
}


void Object::Translate(glm::vec3 translate)
{
    m_ModelMatrix *= glm::translate(translate);
    UpdateModelNormalMatrix();
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


void Object::Scale(glm::vec3 scale)
{
    m_ModelMatrix *= glm::scale(scale);
    UpdateModelNormalMatrix();
}


void Object::Scale(float scale)
{
    Scale(glm::vec3(scale, scale, scale));
}


void Object::Scale(float x, float y, float z)
{
    Scale(glm::vec3(x, y, z));
}

