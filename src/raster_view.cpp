#include "raster_view.hpp"

#include "controller.hpp"
#include "vulkan_engine/vulkan_engine.hpp"
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
  PLOG_DEBUG << "Attaching RasterView layer";
  m_AppHandle = app;
  m_VulkanEngine = new VulkanEngine();
}

void RasterView::OnDetach() { delete m_VulkanEngine; }

void RasterView::OnUpdate() {
  // Check if scene has changed
  auto scene = m_AppHandle->GetActiveScene();
  if (scene &&
      (!m_CurrentScene || (m_CurrentScene && scene->m_SceneID != m_CurrentScene->m_SceneID))) {
    PLOG_VERBOSE << "Active scene changed in RasterView to \"" << scene->GetSceneName()
                 << "\" (Scene ID: " << scene->m_SceneID << ")";
    m_CurrentScene = scene;

    // Update VulkanEngine with new scene
    m_VulkanEngine->SetScene(scene);

    // Update controller with new scene's camera
    Controller::GetInstance()->SetActiveCamera(scene->GetCamera());
  }
}

void RasterView::OnUIRender() {
  // No padding on viewports
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  {
    ImGui::Begin("Rasterized Viewport");
    {
      m_ViewportFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
      if (m_ViewportFocused) {
        m_AppHandle->m_FocusedWindow = Application::RasterizedViewport;
      }

      ImGui::BeginChild("Rasterized");
      {
        m_ViewportHovered = ImGui::IsWindowHovered();

        ImVec2 newSize = ImGui::GetContentRegionAvail();
        if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y) {
          OnResize(newSize);
        }

        m_VulkanEngine->DrawFrame();

        // Wait until the descriptor set for the viewport image is created
        // This could be a source of latency later on -- might be better to
        // add multiple images here as well to allow simultaneous
        // rendering/displaying
        vkDeviceWaitIdle(vke::Device);

        // Note: we flip the image vertically to match Vulkan convention!
        ImGui::Image(m_VulkanEngine->GetImageDescriptorSets()[vke::MainWindowData.FrameIndex],
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

void RasterView::TakeScreenshot() {
  uint32_t width = static_cast<int>(m_ViewportSize.x);
  uint32_t height = static_cast<int>(m_ViewportSize.y);

  auto pixels = m_VulkanEngine->SaveFramebuffer();
  std::vector<uint32_t> out = FlipImageVertically(pixels, width, height);

  // Save screenshot image to file
  WriteImageToFile("output/" + GetDateTimeStr() + "_raster.png", width, height,
                   SDL_PIXELFORMAT_RGBA32, out);
}

// ===================================
// === RasterView specific methods ===
// ===================================

void RasterView::OnResize(ImVec2 newSize) {
  PLOG_DEBUG << "Resizing raster viewport to " << newSize.x << " x " << newSize.y;

  m_ViewportSize = newSize;
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