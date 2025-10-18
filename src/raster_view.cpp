#include "raster_view.hpp"

#include "vulkan_engine/vulkan_utils.hpp"

#include <SDL3/SDL_pixels.h>
#include <plog/Log.h>

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
  m_VulkanEngine = new VulkanEngine();
}

void RasterView::OnDetach() {
  delete m_VulkanEngine;
}

void RasterView::OnUpdate() {}

void RasterView::OnUIRender() {
  // No padding on viewports
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  {
    ImGui::Begin("Rasterized Viewport");
    {
      m_ViewportFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

      ImGui::BeginChild("Rasterized");
      {
        m_ViewportHovered = ImGui::IsWindowHovered();

        ImVec2 newSize = ImGui::GetContentRegionAvail();
        if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y) {
          OnResize(newSize);
        }

        // // m_Scene->VkDraw(*m_Camera);

        // // Wait until the descriptor set for the viewport image is created
        // // This could be a source of latency later on -- might be better to
        // // add multiple images here as well to allow simultaneous
        // // rendering/displaying
        // vkDeviceWaitIdle(vke::Device);

        // // Note: we flip the image vertically to match Vulkan convention!
        // ImGui::Image(m_VulkanEngine.GetImageDescriptorSets()[vke::MainWindowData.FrameIndex],
        //              m_ViewportSize, ImVec2(0, 1), ImVec2(1, 0));
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

  auto pixels = m_VulkanEngine->SaveFramebuffer();
  std::vector<uint32_t> out = FlipImageVertically(pixels, width, height);

  // Save cpt image to file
  WriteImageToFile("output/" + GetDateTimeStr() + "_raster.png", width, height,
                   SDL_PIXELFORMAT_RGBA32, out);
}

// ===================================
// === RasterView specific methods ===
// ===================================

void RasterView::OnResize(ImVec2 newSize) {
  m_ViewportSize = newSize;
  PLOG_INFO << "Resizing raster viewport to " << m_ViewportSize.x << " x "
             << m_ViewportSize.y;
  m_VulkanEngine->OnResize(m_ViewportSize);

  ImVec2 mainWindowPos = ImGui::GetMainViewport()->Pos;
  ImVec2 viewportPos = ImGui::GetWindowPos();
  ImVec2 rPos = ImVec2(viewportPos.x - mainWindowPos.x, viewportPos.y - mainWindowPos.y);
  ImVec2 minR = ImGui::GetWindowContentRegionMin();
  ImVec2 maxR = ImGui::GetWindowContentRegionMax();
  // m_Camera->m_ViewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
  // m_Camera->m_ViewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
  // m_Camera->UpdateProjectionMatrix(static_cast<int>(m_ViewportSize.x),
  //                                  static_cast<int>(m_ViewportSize.y));

  // m_Scene->VkResize(m_ViewportSize, m_ViewportFramebuffers);
}