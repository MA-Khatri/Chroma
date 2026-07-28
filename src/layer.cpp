#include "layer.hpp"

#include <imgui_internal.h>
#include <implot.h>

#include <chrono>
#include <ctime>

std::string GetDateTimeStr() {
  // Get current time as time_t
  auto now = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(now);

  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif

  // Format the time string
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm);

  return std::string(buffer);
}

void Layer::CommonDebug(Application *app, std::shared_ptr<Camera> camera) {
  ImGui::Text("Frame Time: %.3f ms/frame (%.1f FPS)", m_FrameTimes.GetLastItem(),
              m_FrameRates.GetLastItem());

  if (m_IncludeFrameRateGraph) {
    // ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 0));
    ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0, 0, 0, 0));
    ImPlot::SetNextAxisToFit(ImAxis_X1);
    ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 300);
    if (ImPlot::BeginPlot("##FrameRateGraph", ImVec2(-1, 150))) {
      // ImPlot::SetupAxes("", "FPS");
      // ImPlot::SetupAxisTicks(ImAxis_X1, 0, 1000, 11);
      ImPlot::PlotLine("##FrameRate", m_FrameGraphX.data(), m_FrameRates.GetItems().data(),
                       m_FrameGraphStorageCount);
      ImPlot::EndPlot();
    }
  }

  auto viewportSize = camera->GetViewportSize();
  ImGui::Text("Viewport Size :  %.1i x %.1i ", (int)viewportSize.x, (int)viewportSize.y);

  if (ImGui::Button("Take Screenshot")) {
    TakeScreenshot();
  }
}

void Layer::WrapMouseWithinRect(SDL_Window *window, const ImVec2 &rectMin, const ImVec2 &rectMax,
                                bool isActiveViewport, float edgeThreshold) {
  bool isDragging = ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
                    ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                    ImGui::IsMouseDragging(ImGuiMouseButton_Right);

  // Latch whether this drag "started" with the mouse actually over the viewport content.
  if (!m_WasDragging && isDragging) {
    ImVec2 mp = ImGui::GetMousePos();
    m_DragOriginatedInViewport =
        mp.x > rectMin.x && mp.x < rectMax.x && mp.y > rectMin.y && mp.y < rectMax.y;
  }
  m_WasDragging = isDragging;

  if (!window || !m_MouseWrapEnabled || !isActiveViewport || !isDragging ||
      !m_DragOriginatedInViewport || GImGui->MovingWindow != nullptr) {
    return;
  }

  int windowX = 0;
  int windowY = 0;
  SDL_GetWindowPosition(window, &windowX, &windowY);

  const ImVec2 mousePos = ImGui::GetMousePos();
  if (mousePos.x <= rectMin.x + edgeThreshold) {
    SDL_WarpMouseInWindow(window, static_cast<int>(rectMax.x - edgeThreshold - windowX),
                          static_cast<int>(mousePos.y - windowY));
  } else if (mousePos.x >= rectMax.x - edgeThreshold) {
    SDL_WarpMouseInWindow(window, static_cast<int>(rectMin.x + edgeThreshold - windowX),
                          static_cast<int>(mousePos.y - windowY));
  } else if (mousePos.y <= rectMin.y + edgeThreshold) {
    SDL_WarpMouseInWindow(window, static_cast<int>(mousePos.x - windowX),
                          static_cast<int>(rectMax.y - edgeThreshold - windowY));
  } else if (mousePos.y >= rectMax.y - edgeThreshold) {
    SDL_WarpMouseInWindow(window, static_cast<int>(mousePos.x - windowX),
                          static_cast<int>(rectMin.y + edgeThreshold - windowY));
  }
}