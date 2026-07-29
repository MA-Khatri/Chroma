#include "raster_view.hpp"

#include "camera.hpp"
#include "controller.hpp"
#include "vulkan_engine/vulkan_engine.hpp"
#include "vulkan_engine/vulkan_utils.hpp"

#include <SDL3/SDL_pixels.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <plog/Log.h>

RasterView::RasterView(std::string name) : m_ViewportName(name) {
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
  m_WindowHandle = app->GetWindowHandle();
  m_VulkanEngine = new VulkanEngine();
}

void RasterView::OnDetach() { delete m_VulkanEngine; }

void RasterView::OnUpdate() {
  // Check if scene has changed
  auto scene = m_AppHandle->GetActiveScene();
  if (scene &&
      (!m_CurrentScene || (m_CurrentScene && scene->m_SceneID != m_CurrentScene->m_SceneID))) {
    PLOG_DEBUG << "Active scene changed in RasterView to \"" << scene->GetSceneName()
               << "\" (Scene ID: " << scene->m_SceneID << ")";
    m_CurrentScene = scene;

    // Update VulkanEngine with new scene
    m_VulkanEngine->SetScene(scene);

    // TODO: We need to reset the active camera and callback if camera changes
    // Update controller with new scene's camera
    Controller::GetInstance()->SetActiveCamera(scene->GetCamera());

    // Register double click callback
    scene->GetCamera()->GetCameraController()->RegisterDoubleClickCallback(
        [this](glm::vec2 clickPos) -> glm::vec3 {
          return m_VulkanEngine->GetClosestDepth(clickPos);
        });
  }

  // Update frame rate/time
  ImGuiIO io = ImGui::GetIO();

  float frame_time = io.DeltaTime * 1000.0f;
  float frame_rate = 1.0f / io.DeltaTime;

  m_FrameTimes.Add(frame_time);
  m_FrameRates.Add(frame_rate);
}

void RasterView::OnUIRender() {
  // No padding on viewports
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  {
    ImGui::Begin(m_ViewportName.c_str());
    {
      m_WindowID = ImGui::GetID(m_ViewportName.c_str());
      m_ViewportFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
      if (m_ViewportFocused) {
        m_AppHandle->m_FocusedWindow = m_WindowID;
      }
      m_ViewportHovered = false;

      ImGui::BeginChild("Rasterized");
      if (m_ViewportFocused || m_AppHandle->m_FocusedWindow == m_WindowID) {
        m_ViewportHovered = ImGui::IsWindowHovered();

        ImVec2 childMin = ImGui::GetCursorScreenPos();
        ImVec2 childSize = ImGui::GetContentRegionAvail();
        ImVec2 childMax = ImVec2(childMin.x + childSize.x, childMin.y + childSize.y);

        ImVec2 mp = ImGui::GetMousePos();
        const float buffer = 12.0f;
        if (mp.x > childMin.x + buffer && mp.x < childMax.x - buffer &&
            mp.y > childMin.y + buffer && mp.y < childMax.y - buffer) {
          m_CurrentScene->GetCamera()->SetControllerActive(true);
        } else {
          m_CurrentScene->GetCamera()->SetControllerActive(false);
        }

        WrapMouseWithinRect(m_WindowHandle, childMin, childMax, m_ViewportFocused);

        ImVec2 newSize = childSize;
        if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y) {
          m_CurrentScene->GetCamera()->SetViewportBounds(glm::vec2(childMin.x, childMin.y),
                                                         glm::vec2(childMax.x, childMax.y));
          OnResize(newSize);
        }

        m_VulkanEngine->DrawFrame();

        // Wait until the descriptor set for the viewport image is created
        // This could be a source of latency later on -- might be better to
        // add multiple images here as well to allow simultaneous
        // rendering/displaying
        vkDeviceWaitIdle(vke::Device);

        // Note: we flip the image vertically to match Vulkan convention!
        ImGui::Image(
            (ImTextureID)m_VulkanEngine->GetImageDescriptorSets()[vke::MainWindowData.FrameIndex],
            m_ViewportSize, ImVec2(0, 1), ImVec2(1, 0));
      }
      ImGui::EndChild();
    }
    ImGui::End();
  }
  // Add back in padding for non-viewport ImGui
  ImGui::PopStyleVar();

  ImGui::Begin("Debug Panel");
  if (m_ViewportFocused || m_AppHandle->m_FocusedWindow == m_WindowID) {
    Layer::CommonDebug(m_AppHandle, m_CurrentScene->GetCamera());
    m_CurrentScene->GetCamera()->GetGuiElements();
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

void RasterView::OnResize(ImVec2 newSize) {
  PLOG_VERBOSE << "Resizing raster viewport to " << newSize.x << " x " << newSize.y;

  m_ViewportSize = newSize;
  m_VulkanEngine->OnResize(m_ViewportSize);

  // Note: Camera resizing is done in OnUIRender
}