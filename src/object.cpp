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