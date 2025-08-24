#include "raster_view.hpp"

#include "vulkan_engine/vulkan_utils.hpp"

#include <SDL3/SDL_pixels.h>

RasterView::RasterView() {
  // TODO?
}

RasterView::~RasterView() {
  // TODO?
}

// ==============================
// === Standard layer methods ===
// ==============================
void RasterView::OnAttach(Application *app) {
  m_AppHandle = app;
  InitVulkan();
}

void RasterView::OnDetach() { CleanupVulkan(); }

void RasterView::OnUpdate() {}

void RasterView::OnUIRender() {
  // No padding on viewports
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  {
    ImGui::Begin("Rasterized Viewport");
    {
      m_ViewportFocused =
          ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

      ImGui::BeginChild("Rasterized");
      {
        m_ViewportHovered = ImGui::IsWindowHovered();

        ImVec2 newSize = ImGui::GetContentRegionAvail();
        if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y) {
          OnResize(newSize);
        }

        // m_Scene->VkDraw(*m_Camera);

        // Wait until the descriptor set for the viewport image is created
        // This could be a source of latency later on -- might be better to
        // add multiple images here as well to allow simultaneous
        // rendering/displaying
        vkDeviceWaitIdle(vke::Device);

        // Note: we flip the image vertically to match Vulkan convention!
        ImGui::Image(
            m_ViewportImageDescriptorSets[vke::MainWindowData.FrameIndex],
            m_ViewportSize, ImVec2(0, 1), ImVec2(1, 0));
      }
      ImGui::EndChild();
    }
    ImGui::End();
  }
  // Add back in padding for non-viewport ImGui
  ImGui::PopStyleVar();

  ImGui::Begin("Debug Panel");
  {
    // TODO?
  }
  ImGui::End();
}

// Screenshot function for Vulkan based partially on:
// https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot/screenshot.cpp
void RasterView::TakeScreenshot() {
  uint32_t width = static_cast<int>(m_ViewportSize.x);
  uint32_t height = static_cast<int>(m_ViewportSize.y);

  // Create a temporary (capture) image to store screenshot data
  VkImage cptImage;
  VkDeviceMemory cptImageMemory;
  vke::CreateImage(width, height, 1, VK_SAMPLE_COUNT_1_BIT,
                  VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR,
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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
  vkGetImageSubresourceLayout(vke::Device, cptImage, &subResource,
                              &subResourceLayout);

  // Copy cpt image to host
  const char *data;
  vkMapMemory(vke::Device, cptImageMemory, 0, VK_WHOLE_SIZE, 0, (void **)&data);
  data += subResourceLayout.offset;

  // Determine if we need to swizzle
  std::vector<VkFormat> formatsBGR = {VK_FORMAT_B8G8R8A8_SRGB,
                                      VK_FORMAT_B8G8R8A8_UNORM,
                                      VK_FORMAT_B8G8R8A8_SNORM};
  bool swizzle =
      (std::find(formatsBGR.begin(), formatsBGR.end(),
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

  std::vector<uint32_t> out = FlipImageVertically(pixels, width, height);

  // Save cpt image to file
  WriteImageToFile("output/" + GetDateTimeStr() + "_raster.png", width, height,
                   SDL_PIXELFORMAT_RGBA32, out);

  // Cleanup cpt image
  vkUnmapMemory(vke::Device, cptImageMemory);
  vkFreeMemory(vke::Device, cptImageMemory, nullptr);
  vkDestroyImage(vke::Device, cptImage, nullptr);
}

// ===================================
// === RasterView specific methods ===
// ===================================

void RasterView::InitVulkan() {
  m_MSAASampleCount = vke::MaxMSAASamples;

  // Set up viewport rendering
  vke::CreateRenderPass(m_MSAASampleCount, m_ViewportRenderPass);
  vke::CreateViewportSampler(&m_ViewportSampler);

  vke::CreateColorResources(static_cast<uint32_t>(m_ViewportSize.x),
                           static_cast<uint32_t>(m_ViewportSize.y),
                           m_MSAASampleCount, m_ColorImage, m_ColorImageMemory,
                           m_ColorImageView);
  vke::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x),
                           static_cast<uint32_t>(m_ViewportSize.y),
                           m_MSAASampleCount, m_DepthImage, m_DepthImageMemory,
                           m_DepthImageView);
  CreateViewportImagesAndFramebuffers();
  CreateViewportImageDescriptorSets();
}

void RasterView::CleanupVulkan() {
  vkDestroySampler(vke::Device, m_ViewportSampler, nullptr);

  DestroyColorResources();
  DestroyDepthResources();

  DestroyViewportImageDescriptorSets();
  DestroyViewportImagesAndFramebuffers();

  vkDestroyRenderPass(vke::Device, m_ViewportRenderPass, nullptr);
}

void RasterView::OnResize(ImVec2 newSize) { 
  m_ViewportSize = newSize;

  ImVec2 mainWindowPos = ImGui::GetMainViewport()->Pos;
  ImVec2 viewportPos = ImGui::GetWindowPos();
  ImVec2 rPos =
      ImVec2(viewportPos.x - mainWindowPos.x, viewportPos.y - mainWindowPos.y);
  ImVec2 minR = ImGui::GetWindowContentRegionMin();
  ImVec2 maxR = ImGui::GetWindowContentRegionMax();
  // m_Camera->m_ViewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
  // m_Camera->m_ViewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
  // m_Camera->UpdateProjectionMatrix(static_cast<int>(m_ViewportSize.x),
  //                                  static_cast<int>(m_ViewportSize.y));

  // Before re-creating images, we MUST wait for device to be done using them
  vkDeviceWaitIdle(vke::Device);

  // Cleanup previous
  DestroyColorResources();
  DestroyDepthResources();
  DestroyViewportImagesAndFramebuffers();
  DestroyViewportImageDescriptorSets();

  // Recreate new
  vke::CreateColorResources(static_cast<uint32_t>(m_ViewportSize.x),
                           static_cast<uint32_t>(m_ViewportSize.y),
                           m_MSAASampleCount, m_ColorImage, m_ColorImageMemory,
                           m_ColorImageView);
  vke::CreateDepthResources(static_cast<uint32_t>(m_ViewportSize.x),
                           static_cast<uint32_t>(m_ViewportSize.y),
                           m_MSAASampleCount, m_DepthImage, m_DepthImageMemory,
                           m_DepthImageView);
  CreateViewportImagesAndFramebuffers();
  CreateViewportImageDescriptorSets();

  // m_Scene->VkResize(m_ViewportSize, m_ViewportFramebuffers);
}

void RasterView::CreateViewportImagesAndFramebuffers() {
  vke::CreateViewportImages(vke::ImageCount, m_ViewportSize, m_ViewportImages,
                           m_ViewportImagesDeviceMemory);
  vke::CreateViewportImageViews(m_ViewportImages, m_ViewportImageViews);
  m_ViewportFramebuffers.resize(vke::ImageCount);
  for (uint32_t i = 0; i < vke::ImageCount; i++) {
    vke::CreateFrameBuffer(
        std::vector<VkImageView>{m_ColorImageView, m_DepthImageView,
                                 m_ViewportImageViews[i]},
        m_ViewportRenderPass, m_ViewportSize, m_ViewportFramebuffers[i]);
  }
}

void RasterView::DestroyViewportImagesAndFramebuffers() {
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

void RasterView::CreateViewportImageDescriptorSets() {
  for (uint32_t i = 0; i < vke::ImageCount; i++) {
    m_ViewportImageDescriptorSets.push_back(
        (VkDescriptorSet)ImGui_ImplVulkan_AddTexture(
            m_ViewportSampler, m_ViewportImageViews[i],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
  }
}

void RasterView::DestroyViewportImageDescriptorSets() {
  for (auto &descriptorSet : m_ViewportImageDescriptorSets) {
    ImGui_ImplVulkan_RemoveTexture(descriptorSet);
  }

  m_ViewportImageDescriptorSets.clear();
}

void RasterView::DestroyColorResources() {
  vkDestroyImageView(vke::Device, m_ColorImageView, nullptr);
  vkDestroyImage(vke::Device, m_ColorImage, nullptr);
  vkFreeMemory(vke::Device, m_ColorImageMemory, nullptr);
}

void RasterView::DestroyDepthResources() {
  vkDestroyImageView(vke::Device, m_DepthImageView, nullptr);
  vkDestroyImage(vke::Device, m_DepthImage, nullptr);
  vkFreeMemory(vke::Device, m_DepthImageMemory, nullptr);
}