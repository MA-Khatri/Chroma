#pragma once

#include <imgui.h>
#include <memory>
#include <vulkan/vulkan.h>

#include "layer.hpp"
#include "vulkan_engine/vulkan_engine.hpp"

class RasterView : public Layer {
public:
  RasterView(std::string name);
  ~RasterView();

  virtual void OnAttach(Application *app) final;
  virtual void OnDetach() final;

  virtual void OnUpdate() final;
  virtual void OnRender() final;
  virtual void OnUIRender() final;

  virtual void TakeScreenshot();

protected:
  // (Optional) Child-class hooks
  virtual void OnAttachHook();
  virtual void OnUpdateHook();
  virtual void ControlPanelHook();

  VulkanEngine *m_VulkanEngine;

private:
  // RasterView specific methods
  void InitVulkan();
  void CleanupVulkan();
  void OnResize(ImVec2 min, ImVec2 max);
};
