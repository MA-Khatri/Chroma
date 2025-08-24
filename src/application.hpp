#pragma once

#include <cstdint> // for int64_t
#include <functional>
#include <memory>

#include <SDL3/SDL.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include "layer.hpp"
#include "scene.hpp"

// Forward declarations
class Layer;
class Scene;

// Singleton
class Application {
public:
  static Application *GetInstance() {
    if (s_Instance == nullptr) {
      // Lazy initialize new instance
      s_Instance = new Application();
    }
    return s_Instance;
  }

  void Run();
  void Close();

  void SetMenubarCallback(const std::function<void()> &menubarCallback);
  std::function<void()> GetMenubarCallback();

  void PushLayer(const std::shared_ptr<Layer> &layer);

  SDL_Window *GetWindowHandle() const { return m_WindowHandle; }

  int64_t GetTimeNS(); // Current time in nanoseconds

  enum { RasterizedViewport, RayTracedViewport };
  int m_FocusedWindow = RasterizedViewport;

private:
  static Application *s_Instance;
  Application();
  ~Application();

  // Remove copy constructor and assignment operator
  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;

  void Init();
  void Shutdown();
  void NextFrame();

  SDL_Window *m_WindowHandle;
  std::function<void()> m_MenubarCallback;
  std::vector<std::shared_ptr<Layer>> m_Layers;

  bool m_Running;

  // Time is stored in nanoseconds
  int64_t m_FrameTimeNS;
  int64_t m_TimeStepNS;
  int64_t m_LastFrameTimeNS;
};

constexpr uint64_t SecondsToNanoseconds(double seconds);
constexpr double NanosecondsToSeconds(uint64_t nanoseconds);