#include "layer.hpp"

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

void Layer::WrapMouseWithinRect(SDL_Window *window, const ImVec2 &rectMin, const ImVec2 &rectMax,
                                bool isActiveViewport, float edgeThreshold) {
  if (!window || !m_MouseWrapEnabled || !isActiveViewport ||
      !(ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
        ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
        ImGui::IsMouseDragging(ImGuiMouseButton_Right))) {
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