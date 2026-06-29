#pragma once

#include <imgui.h>
#include <memory>
#include <vulkan/vulkan.h>

#include "layer.hpp"
#include "vulkan_engine/vulkan_engine.hpp"

class RasterView : public Layer {
public:
  RasterView();
  ~RasterView();

  // Standard layer methods
  virtual void OnAttach(Application *app);
  virtual void OnDetach();
  virtual void OnUpdate();
  virtual void OnUIRender();

  virtual void TakeScreenshot();

private:
  // RasterView specific methods
  void InitVulkan();
  void CleanupVulkan();
  void OnResize(ImVec2 newSize);

  // void SceneSetup();

private:
  Application *m_AppHandle;
  SDL_Window *m_WindowHandle;
  VulkanEngine *m_VulkanEngine;

  std::shared_ptr<Scene> m_CurrentScene;

  bool m_ViewportFocused = false;
  bool m_ViewportHovered = false;
  ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);
};
