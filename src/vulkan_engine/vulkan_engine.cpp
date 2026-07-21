#include "vulkan_engine.hpp"
#include "vulkan_utils.hpp"

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

  // Recreate new
  vke::CreateColorResources(static_cast<uint32_t>(m_ViewportSize.x),
                            static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount,
                            m_ColorImage, m_ColorImageMemory, m_ColorImageView);
  vke::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x),
                            static_cast<uint32_t>(m_ViewportSize.y), m_MSAASampleCount,
                            m_DepthImage, m_DepthImageMemory, m_DepthImageView);
  CreateViewportImagesAndFramebuffers();
  CreateViewportImageDescriptorSets();
}

// Screenshot functionality for Vulkan based partially on:
// https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot/screenshot.cpp
std::vector<uint32_t> VulkanEngine::SaveFramebuffer() {
  uint32_t width = static_cast<int>(m_ViewportSize.x);
  uint32_t height = static_cast<int>(m_ViewportSize.y);

  // Create a temporary (capture) image to store screenshot data
  VkImage cptImage;
  VkDeviceMemory cptImageMemory;
  vke::CreateImage(width, height, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_TILING_LINEAR,
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   cptImage, cptImageMemory);

  // Get the current viewport image
  VkImage &srcImage = m_ViewportImages[vke::MainWindowData.FrameIndex];

  // Transition viewport image to transfer src optimal
  vke::TransitionImageLayout(srcImage, vke::MainWindowData.SurfaceFormat.format,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);

  // Copy viewport image to cpt image
  vke::CopyImageToImage(m_ViewportSize, srcImage, cptImage);

  // Transition viewport image back to color attachment optimal
  vke::TransitionImageLayout(srcImage, vke::MainWindowData.SurfaceFormat.format,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);

  // Transition cpt image to transfer src optimal
  vke::TransitionImageLayout(cptImage, VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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

  m_VkScene = std::make_shared<VkScene>(scene, m_MSAASampleCount, m_ViewportRenderPass);
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