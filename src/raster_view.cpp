#include "raster_view.hpp"

#include "camera.hpp"
#include "controller.hpp"
#include "vulkan_engine/vulkan_engine.hpp"
#include "vulkan_engine/vulkan_utils.hpp"

#include <SDL3/SDL_pixels.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <plog/Log.h>

RasterView::RasterView(std::string name) {
  m_ViewportName = name;
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
  m_WindowHandle = app->GetWindowHandle();
  m_VulkanEngine = new VulkanEngine();

  OnAttachHook();

  // Register double click callback
  m_CurrentScene->GetCamera()->GetCameraController()->RegisterDoubleClickCallback(
      [this](glm::vec2 clickPos) -> glm::vec3 {
        return m_VulkanEngine->GetClosestDepth(clickPos);
      });
}

void RasterView::OnDetach() { delete m_VulkanEngine; }

void RasterView::OnUpdate() {
  // Update frame rate/time
  ImGuiIO io = ImGui::GetIO();

  float frame_time = io.DeltaTime * 1000.0f;
  float frame_rate = 1.0f / io.DeltaTime;

  m_FrameTimes.Add(frame_time);
  m_FrameRates.Add(frame_rate);

  // No padding on viewports
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  // Update Viewport info
  ImGui::Begin(m_ViewportName.c_str());
  {
    m_WindowID = ImGui::GetID(m_ViewportName.c_str());
    m_ViewportFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
    if (m_ViewportFocused) {
      m_AppHandle->m_FocusedWindow = m_WindowID;
      Controller::GetInstance()->SetActiveCamera(m_CurrentScene->GetCamera());
    }
    m_ViewportHovered = false;

    if (!ImGui::IsWindowCollapsed()) {
      m_ViewportHovered = ImGui::IsWindowHovered();

      ImVec2 childMin = ImGui::GetCursorScreenPos();
      ImVec2 childSize = ImGui::GetContentRegionAvail();
      ImVec2 childMax = ImVec2(childMin.x + childSize.x, childMin.y + childSize.y);

      ImVec2 mp = ImGui::GetMousePos();
      const float buffer = 12.0f;
      if (mp.x > childMin.x + buffer && mp.x < childMax.x - buffer && mp.y > childMin.y + buffer &&
          mp.y < childMax.y - buffer) {
        m_CurrentScene->GetCamera()->SetControllerActive(true);
      } else {
        m_CurrentScene->GetCamera()->SetControllerActive(false);
      }

      ImVec2 newSize = childSize;
      if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y) {
        OnResize(childMin, childMax);
      }

      WrapMouseWithinRect(m_WindowHandle, childMin, childMax, m_ViewportFocused);
    }
  }
  ImGui::End();
  ImGui::PopStyleVar();

  OnUpdateHook();
}

void RasterView::OnRender() {
  m_VulkanEngine->DrawFrame();
  vkDeviceWaitIdle(vke::Device);
}

void RasterView::OnUIRender() {
  ImGui::Begin(m_ViewportName.c_str());
  if (!ImGui::IsWindowCollapsed()) {
    // Note: we flip the image vertically to match Vulkan convention!
    ImGui::Image(
        (ImTextureID)m_VulkanEngine->GetImageDescriptorSets()[vke::MainWindowData.FrameIndex],
        m_ViewportSize, ImVec2(0, 1), ImVec2(1, 0));
  }
  ImGui::End();

  ImGui::Begin(m_ControlPanelName.c_str());
  if (m_AppHandle->m_FocusedWindow == m_WindowID) {
    Layer::CommonControlPanel(m_AppHandle, m_CurrentScene->GetCamera());
    m_CurrentScene->GetCamera()->GetGuiElements();
    ControlPanelHook();
  }
  ImGui::End();
}

void RasterView::TakeScreenshot() {
  uint32_t width = static_cast<int>(m_ViewportSize.x);
  uint32_t height = static_cast<int>(m_ViewportSize.y);

  auto pixels = m_VulkanEngine->SaveFramebuffer();
  std::vector<uint32_t> out = FlipImageVertically(pixels, width, height);

  // Save screenshot image to file
  WriteImageToFile(GetDateTimeStr() + "_raster.png", width, height, SDL_PIXELFORMAT_RGBA32, out);
}

// ===================================
// === RasterView specific methods ===
// ===================================

void RasterView::OnResize(ImVec2 min, ImVec2 max) {
  m_CurrentScene->GetCamera()->SetViewportBounds(glm::vec2(min.x, min.y), glm::vec2(max.x, max.y));

  ImVec2 newSize = ImVec2(max.x - min.x, max.y - min.y);
  m_ViewportSize = newSize;
  m_VulkanEngine->OnResize(m_ViewportSize);
}

void RasterView::OnAttachHook() {
  m_CurrentScene = std::make_shared<Scene>(CreateTestScene());
  m_VulkanEngine->SetScene(m_CurrentScene);
}

void RasterView::OnUpdateHook() {
  // TODO?
}

void RasterView::ControlPanelHook() {
  // TODO?
}