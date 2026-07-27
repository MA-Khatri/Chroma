#include "vulkan_engine.hpp"
#include "vulkan_utils.hpp"

#include <cstdint>
#include <limits>
#include <plog/Log.h>
#include <vulkan/vulkan_core.h>

/// =============================
/// ====== Public Methods =======
/// =============================

VulkanEngine::VulkanEngine() { InitVulkan(); }

VulkanEngine::~VulkanEngine() { CleanupVulkan(); }

void VulkanEngine::OnResize(ImVec2 newSize) {
  m_ViewportSize = newSize;

  // Before re-creating images, we MUST wait for device to be done using them
  vkDeviceWaitIdle(vke::Device);

  // Cleanup previous
  DestroyColorResources();
  DestroyDepthResources();
  DestroyViewportImagesAndFramebuffers();
  DestroyViewportImageDescriptorSets();

  DestroyPickResources();

  // Recreate new
  vke::CreateColorResources(static_cast<uint32_t>(m_ViewportSize.x),
                            static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount,
                            m_ColorImage, m_ColorImageMemory, m_ColorImageView);
  vke::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x),
                            static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount,
                            m_DepthImage, m_DepthImageMemory, m_DepthImageView);
  CreateViewportImagesAndFramebuffers();
  CreateViewportImageDescriptorSets();

  CreatePickResources();
}

// Screenshot functionality for Vulkan based partially on:
// https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot/screenshot.cpp
std::vector<uint32_t> VulkanEngine::SaveFramebuffer() {
  uint32_t width = static_cast<int>(m_ViewportSize.x);
  uint32_t height = static_cast<int>(m_ViewportSize.y);

  // Create a temporary (capture) image to store screenshot data
  VkImage cptImage;
  VkDeviceMemory cptImageMemory;
  VkFormat cptImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
  vke::CreateImage(width, height, 1, VK_SAMPLE_COUNT_1_BIT, cptImageFormat, VK_IMAGE_TILING_LINEAR,
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   cptImage, cptImageMemory);
  vke::TransitionImageLayout(cptImage, cptImageFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);

  // Get the current viewport image
  VkImage &srcImage = m_ViewportImages[vke::MainWindowData.FrameIndex];

  // Transition viewport image to transfer src optimal
  vke::TransitionImageLayout(srcImage, vke::MainWindowData.SurfaceFormat.format,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);

  // Copy viewport image to cpt image
  vke::CopyImageToImage(m_ViewportSize, srcImage, cptImage);

  // Transition viewport image back to color attachment optimal
  vke::TransitionImageLayout(srcImage, vke::MainWindowData.SurfaceFormat.format,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

  // Transition cpt image to transfer src optimal
  vke::TransitionImageLayout(cptImage, cptImageFormat, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);

  // Get layout of the image (including row pitch)
  VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  VkSubresourceLayout subResourceLayout;
  vkGetImageSubresourceLayout(vke::Device, cptImage, &subResource, &subResourceLayout);

  // Copy cpt image to host
  const char *data;
  vkMapMemory(vke::Device, cptImageMemory, 0, VK_WHOLE_SIZE, 0, (void **)&data);
  data += subResourceLayout.offset;

  // Determine if we need to swizzle
  std::vector<VkFormat> formatsBGR = {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM,
                                      VK_FORMAT_B8G8R8A8_SNORM};
  bool swizzle = (std::find(formatsBGR.begin(), formatsBGR.end(),
                            vke::MainWindowData.SurfaceFormat.format) != formatsBGR.end());

  // Save image to vector with proper format
  std::vector<uint32_t> pixels;
  pixels.resize(width * height);
  for (uint32_t y = 0; y < height; y++) {
    uint32_t *row = (uint32_t *)data;
    for (uint32_t x = 0; x < width; x++) {
      uint32_t pixelID = x + y * width;
      uint32_t pixel = 0;
      if (swizzle) {
        uint8_t b0, b1, b2, b3;
        uint32_t color = *row;
        b0 = 0xff; // alpha
        b1 = (color >> 0) & 0xff;
        b2 = (color >> 8) & 0xff;
        b3 = (color >> 16) & 0xff;

        pixel = b0 << 24 | b1 << 16 | b2 << 8 | b3 << 0;
      } else {
        pixel = *row | 0xff000000;
      }
      pixels[pixelID] = pixel;
      row++;
    }
    data += subResourceLayout.rowPitch;
  }

  // Cleanup cpt image
  vkUnmapMemory(vke::Device, cptImageMemory);
  vkFreeMemory(vke::Device, cptImageMemory, nullptr);
  vkDestroyImage(vke::Device, cptImage, nullptr);

  return pixels;
}

void VulkanEngine::SetScene(std::shared_ptr<Scene> scene) {
  // Check if we already have a VkScene for this Scene
  // TODO: What happens if the Scene is modified?
  if (m_VkScenes.find(scene->m_SceneID) != m_VkScenes.end()) {
    m_VkScene = m_VkScenes[scene->m_SceneID];
    return;
  }

  m_VkScene =
      std::make_shared<VkScene>(scene, m_MSAASampleCount, m_ViewportRenderPass, m_PickRenderPass);
  m_VkScenes.insert({scene->m_SceneID, m_VkScene});
}

void VulkanEngine::DrawFrame() {
  VkCommandBuffer commandBuffer = vke::GetGraphicsCommandBuffer();

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = m_ViewportRenderPass;
  renderPassInfo.framebuffer = m_ViewportFramebuffers[vke::MainWindowData.FrameIndex];
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = {static_cast<uint32_t>(m_ViewportSize.x),
                                      static_cast<uint32_t>(m_ViewportSize.y)};

  std::array<VkClearValue, 2> clearValues{};
  auto cc = m_VkScene->GetBaseScene()->GetClearColor();
  clearValues[0].color = {{cc.x, cc.y, cc.z, 1.0f}};
  clearValues[1].depthStencil = {1.0f, 0};
  renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
  renderPassInfo.pClearValues = clearValues.data();

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  // Need to set the viewport and scissor since they are dynamic
  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = m_ViewportSize.x;
  viewport.height = m_ViewportSize.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {static_cast<uint32_t>(m_ViewportSize.x),
                    static_cast<uint32_t>(m_ViewportSize.y)};
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  m_VkScene->Draw(commandBuffer, m_ViewportSize);

  vkCmdEndRenderPass(commandBuffer);

  vke::FlushGraphicsCommandBuffer(commandBuffer);
}

void VulkanEngine::DrawPickFrame() {
  VkCommandBuffer commandBuffer = vke::GetGraphicsCommandBuffer();

  VkRenderPassBeginInfo pickRenderPassInfo{};
  pickRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  pickRenderPassInfo.renderPass = m_PickRenderPass;
  pickRenderPassInfo.framebuffer = m_PickFramebuffer;
  pickRenderPassInfo.renderArea.offset = {0, 0};
  pickRenderPassInfo.renderArea.extent = {static_cast<uint32_t>(m_ViewportSize.x),
                                          static_cast<uint32_t>(m_ViewportSize.y)};

  VkClearValue pickDepthClearValue{};
  pickDepthClearValue.depthStencil = {1.0f, 0};
  pickRenderPassInfo.clearValueCount = 1;
  pickRenderPassInfo.pClearValues = &pickDepthClearValue;

  vkCmdBeginRenderPass(commandBuffer, &pickRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  // Need to set the viewport and scissor since they are dynamic
  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = m_ViewportSize.x;
  viewport.height = m_ViewportSize.y;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {static_cast<uint32_t>(m_ViewportSize.x),
                    static_cast<uint32_t>(m_ViewportSize.y)};
  vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

  m_VkScene->DrawPick(commandBuffer, m_ViewportSize);

  vkCmdEndRenderPass(commandBuffer);

  vke::FlushGraphicsCommandBuffer(commandBuffer);
}

std::tuple<VkRect2D, std::vector<float>> VulkanEngine::GetDepthBuffer(VkRect2D rect) {
  if (m_PickDepthImageFormat != VK_FORMAT_D32_SFLOAT) {
    PLOG_ERROR << "GetDepthBuffer(): Expected m_PickDepthImageFormat to be VK_FORMAT_D32_SFLOAT, "
                  "but it is "
               << m_PickDepthImageFormat << "!";
    return {};
  }

  // Draw new depth buffer
  DrawPickFrame();

  // Copy region of depth buffer
  vke::TransitionImageLayout(m_PickDepthImage, m_PickDepthImageFormat,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);

  const VkDeviceSize bytesPerTexel = sizeof(float); // D32_SFLOAT is a single 32-bit float
  const VkDeviceSize bufferSize = rect.extent.width * rect.extent.height * bytesPerTexel;

  // Host-visible staging buffer to receive the region
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  vke::CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    stagingBuffer, stagingBufferMemory);

  // Record and submit the image -> buffer copy
  // VkCommandBuffer commandBuffer = vke::GetTransferCommandBuffer();
  VkCommandBuffer commandBuffer = vke::GetGraphicsCommandBuffer();

  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;   // tightly packed
  region.bufferImageHeight = 0; // tightly packed
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {rect.offset.x, rect.offset.y, 0};
  region.imageExtent = {rect.extent.width, rect.extent.height, 1};

  vkCmdCopyImageToBuffer(commandBuffer, m_PickDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         stagingBuffer, 1, &region);

  // vke::FlushTransferCommandBuffer(commandBuffer);
  vke::FlushGraphicsCommandBuffer(commandBuffer);

  // Put the depth image back into a renderable layout for next frame
  vke::TransitionImageLayout(m_PickDepthImage, m_PickDepthImageFormat,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

  // Copy out of the mapped staging buffer into the result vector
  void *mappedData;
  vkMapMemory(vke::Device, stagingBufferMemory, 0, bufferSize, 0, &mappedData);

  std::vector<float> result(rect.extent.width * rect.extent.height);
  memcpy(result.data(), mappedData, bufferSize);

  vkUnmapMemory(vke::Device, stagingBufferMemory);

  vkDestroyBuffer(vke::Device, stagingBuffer, nullptr);
  vkFreeMemory(vke::Device, stagingBufferMemory, nullptr);

  return {rect, result};
}

std::tuple<VkRect2D, std::vector<float>>
VulkanEngine::GetDepthBuffer(int startX, int startY, unsigned int extentX, unsigned int extentY) {
  VkRect2D rect{};
  rect.offset = {startX, startY};
  rect.extent = {extentX, extentY};

  return GetDepthBuffer(rect);
}

std::tuple<VkRect2D, std::vector<float>> VulkanEngine::GetDepthBuffer() {
  return GetDepthBuffer(0, 0, m_ViewportSize.x, m_ViewportSize.y);
}

std::vector<float> VulkanEngine::GetPickDepth(int cx, int cy) {
  if (m_PickDepthImageFormat != VK_FORMAT_D32_SFLOAT) {
    PLOG_ERROR << "GetPickDepth(): Expected m_PickDepthImageFormat to be VK_FORMAT_D32_SFLOAT, "
                  "but it is "
               << m_PickDepthImageFormat << "!";
    return {};
  }

  // Clamp the pick region to the viewport bounds
  const int viewportWidth = static_cast<int>(m_ViewportSize.x);
  const int viewportHeight = static_cast<int>(m_ViewportSize.y);
  const int half = m_PickDiameter / 2;

  // Flip vertically to fit Vulkan convention
  int cyf = m_ViewportSize.y - cy;

  int startX = std::clamp(cx - half, 0, viewportWidth);
  int startY = std::clamp(cyf - half, 0, viewportHeight);
  int endX = std::clamp(cx - half + static_cast<int>(m_PickDiameter), 0, viewportWidth);
  int endY = std::clamp(cyf - half + static_cast<int>(m_PickDiameter), 0, viewportHeight);

  const uint32_t extentX = static_cast<uint32_t>(endX - startX);
  const uint32_t extentY = static_cast<uint32_t>(endY - startY);

  if (extentX == 0 || extentY == 0) {
    PLOG_ERROR << "GetPickDepth(): Requested pick center (" << cx << ", " << cy
               << ") lies entirely outside the viewport!";
    return {};
  }

  const VkDeviceSize bytesPerTexel = sizeof(float); // D32_SFLOAT is a single 32-bit float
  const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(extentX) * extentY * bytesPerTexel;

  if (bufferSize > m_ReadbackSize) {
    PLOG_ERROR << "GetPickDepth(): Requested region (" << bufferSize
               << " bytes) exceeds readback buffer capacity (" << m_ReadbackSize << " bytes)!";
    return {};
  }

  // Draw new depth buffer
  DrawPickFrame();

  // Copy region of depth buffer
  vke::TransitionImageLayout(m_PickDepthImage, m_PickDepthImageFormat,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);

  VkCommandBuffer commandBuffer = vke::GetGraphicsCommandBuffer();

  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;   // tightly packed
  region.bufferImageHeight = 0; // tightly packed
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {startX, startY, 0};
  region.imageExtent = {extentX, extentY, 1};

  vkCmdCopyImageToBuffer(commandBuffer, m_PickDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         m_PickDepthReadbackBuffer, 1, &region);

  // Waits on a fence internally, so the copy is guaranteed complete before we
  // touch m_PickDepthMappedReadback below. HOST_COHERENT means no explicit
  // invalidate needed.
  vke::FlushGraphicsCommandBuffer(commandBuffer);

  // Put the depth image back into a renderable layout for next frame
  vke::TransitionImageLayout(m_PickDepthImage, m_PickDepthImageFormat,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

  // Copy out of the persistently mapped readback buffer into the result vector
  std::vector<float> result(static_cast<size_t>(extentX) * extentY);
  memcpy(result.data(), m_PickDepthMappedReadback, bufferSize);

  return result;
}

glm::vec3 VulkanEngine::GetClosestDepth(glm::vec2 clickPosn) {
  constexpr int halfPick = m_PickDiameter / 2;

  std::vector<float> pickDepth = GetPickDepth(clickPosn.x, clickPosn.y);
  if (pickDepth.size() == 0)
    return glm::vec3(std::numeric_limits<float>::quiet_NaN());

  glm::vec2 closestDepthOffset(0, 0);
  float closestOffsetDist = 10000.0f;
  float closestDepth = 1.0f;
  for (int j = 0; j < m_PickDiameter; j++) {
    unsigned int rowOffset = j * m_PickDiameter;
    for (int i = 0; i < m_PickDiameter; i++) {
      unsigned int idx = rowOffset + i;
      float cdepth = pickDepth[idx];
      if (cdepth < 1.0f) { // i.e., valid depth
        glm::vec2 offset(i - halfPick, j - halfPick);
        float offsetDist = glm::length2(offset);
        if (offsetDist < closestOffsetDist) {
          closestOffsetDist = offsetDist;
          closestDepthOffset = offset;
          closestDepth = cdepth;
        }
      }
    }
  }

  return glm::vec3(clickPosn + closestDepthOffset, closestDepth);
}

/// =============================
/// ====== Private Methods ======
/// =============================

void VulkanEngine::InitVulkan() {
  m_MSAASampleCount =
      vke::MaxMSAASamples > VK_SAMPLE_COUNT_4_BIT ? VK_SAMPLE_COUNT_4_BIT : vke::MaxMSAASamples;

  // Set up viewport rendering
  vke::CreateRenderPass(m_MSAASampleCount, m_ViewportRenderPass);
  vke::CreateViewportSampler(&m_ViewportSampler);

  vke::CreateColorResources(static_cast<uint32_t>(m_ViewportSize.x),
                            static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount,
                            m_ColorImage, m_ColorImageMemory, m_ColorImageView);
  vke::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x),
                            static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount,
                            m_DepthImage, m_DepthImageMemory, m_DepthImageView);
  CreateViewportImagesAndFramebuffers();
  CreateViewportImageDescriptorSets();

  vke::CreatePickRenderPass(m_PickRenderPass);
  CreatePickResources();

  // Ensure device is idle before finishing init
  vkDeviceWaitIdle(vke::Device);
}

void VulkanEngine::CleanupVulkan() {
  vkDestroySampler(vke::Device, m_ViewportSampler, nullptr);

  DestroyColorResources();
  DestroyDepthResources();

  DestroyViewportImageDescriptorSets();
  DestroyViewportImagesAndFramebuffers();

  vkDestroyRenderPass(vke::Device, m_ViewportRenderPass, nullptr);

  DestroyPickResources();
  vkDestroyRenderPass(vke::Device, m_PickRenderPass, nullptr);
}

void VulkanEngine::CreateViewportImagesAndFramebuffers() {
  vke::CreateViewportImages(vke::ImageCount, m_ViewportSize, m_ViewportImages,
                            m_ViewportImagesDeviceMemory);
  vke::CreateViewportImageViews(m_ViewportImages, m_ViewportImageViews);
  m_ViewportFramebuffers.resize(vke::ImageCount);
  for (uint32_t i = 0; i < vke::ImageCount; i++) {
    vke::CreateFrameBuffer(
        std::vector<VkImageView>{m_ColorImageView, m_DepthImageView, m_ViewportImageViews[i]},
        m_ViewportRenderPass, m_ViewportSize, m_ViewportFramebuffers[i]);
  }
}

void VulkanEngine::DestroyViewportImagesAndFramebuffers() {
  for (uint32_t i = 0; i < vke::ImageCount; i++) {
    vkDestroyFramebuffer(vke::Device, m_ViewportFramebuffers[i], nullptr);
  }

  for (uint32_t i = 0; i < vke::ImageCount; i++) {
    vkDestroyImageView(vke::Device, m_ViewportImageViews[i], nullptr);
    vkDestroyImage(vke::Device, m_ViewportImages[i], nullptr);
    vkFreeMemory(vke::Device, m_ViewportImagesDeviceMemory[i], nullptr);
  }

  // Need to clear the memory vector otherwise we may get errors saying that
  // we are trying to free already freed memory if we call CreateImage() after.
  m_ViewportImagesDeviceMemory.clear();
}

void VulkanEngine::CreateViewportImageDescriptorSets() {
  for (uint32_t i = 0; i < vke::ImageCount; i++) {
    m_ViewportImageDescriptorSets.push_back((VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
        m_ViewportSampler, m_ViewportImageViews[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
  }
}

void VulkanEngine::DestroyViewportImageDescriptorSets() {
  for (auto &descriptorSet : m_ViewportImageDescriptorSets) {
    ImGui_ImplVulkan_RemoveTexture(descriptorSet);
  }

  m_ViewportImageDescriptorSets.clear();
}

void VulkanEngine::DestroyColorResources() {
  vkDestroyImageView(vke::Device, m_ColorImageView, nullptr);
  vkDestroyImage(vke::Device, m_ColorImage, nullptr);
  vkFreeMemory(vke::Device, m_ColorImageMemory, nullptr);
}

void VulkanEngine::DestroyDepthResources() {
  vkDestroyImageView(vke::Device, m_DepthImageView, nullptr);
  vkDestroyImage(vke::Device, m_DepthImage, nullptr);
  vkFreeMemory(vke::Device, m_DepthImageMemory, nullptr);
}

void VulkanEngine::CreatePickResources() {
  // Pick depth image
  m_PickDepthImageFormat = vke::FindDepthFormat();
  vke::CreateImage(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y),
                   1, VK_SAMPLE_COUNT_1_BIT, m_PickDepthImageFormat, VK_IMAGE_TILING_OPTIMAL,
                   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_PickDepthImage, m_PickDepthImageMemory);
  vke::CreateImageView(m_PickDepthImageFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1, m_PickDepthImage,
                       m_PickDepthImageView);
  vke::TransitionImageLayout(m_PickDepthImage, m_PickDepthImageFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

  vke::CreateFrameBuffer({m_PickDepthImageView}, m_PickRenderPass, m_ViewportSize,
                         m_PickFramebuffer);

  // Pick depth readback buffer
  vke::CreateBuffer(m_ReadbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    m_PickDepthReadbackBuffer, m_PickDepthReadbackMemory);
  vkMapMemory(vke::Device, m_PickDepthReadbackMemory, 0, VK_WHOLE_SIZE, 0,
              &m_PickDepthMappedReadback);
}

void VulkanEngine::DestroyPickResources() {
  vkDestroyFramebuffer(vke::Device, m_PickFramebuffer, nullptr);
  vkDestroyImageView(vke::Device, m_PickDepthImageView, nullptr);
  vkDestroyImage(vke::Device, m_PickDepthImage, nullptr);
  vkFreeMemory(vke::Device, m_PickDepthImageMemory, nullptr);

  vkUnmapMemory(vke::Device, m_PickDepthReadbackMemory);
  vkDestroyBuffer(vke::Device, m_PickDepthReadbackBuffer, nullptr);
  vkFreeMemory(vke::Device, m_PickDepthReadbackMemory, nullptr);
}