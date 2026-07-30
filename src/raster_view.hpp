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

  // Standard layer methods
  virtual void OnAttach(Application *app) final;
  virtual void OnDetach() final;
  virtual void OnUpdate() final;
  virtual void OnUIRender() final;

  virtual void TakeScreenshot();

protected:
  // (Optional) Child-class hooks
  virtual void OnAttachExtra();
  virtual void OnUpdateExtra();
  virtual void ControlPanelExtra();

  VulkanEngine *m_VulkanEngine;

private:
  // RasterView specific methods
  void InitVulkan();
  void CleanupVulkan();
  void OnResize(ImVec2 newSize);
};
