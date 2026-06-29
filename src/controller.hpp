#pragma once

#include <functional>
#include <memory>

#include "camera.hpp"

// Singleton; Handles application state and user inputs
class Controller {
public:
  static Controller *GetInstance() {
    if (s_Instance == nullptr) {
      // Lazy initialize new instance
      s_Instance = new Controller();
    }
    return s_Instance;
  }

  void ProcessEvents();

  void Close() { m_Running = false; }
  bool Running() const { return m_Running; }

  void SetMenubarCallback(const std::function<void()> &menubarCallback);
  std::function<void()> GetMenubarCallback();

  void SetActiveCamera(std::shared_ptr<Camera> camera) { m_Camera = camera; }

private:
  static Controller *s_Instance;
  Controller();
  ~Controller();

  // Remove copy constructor and assignment operator
  Controller(const Controller &) = delete;
  Controller &operator=(const Controller &) = delete;

  bool m_Running = true;

  std::function<void()> m_MenubarCallback;

  // The currently active camera instance
  std::shared_ptr<Camera> m_Camera;
};