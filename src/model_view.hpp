#pragma once

#include "camera.hpp"
#include "raster_view.hpp"
#include "scene.hpp"

class ModelView : public RasterView {
public:
  ModelView(std::string name) : RasterView(name) {};
  ~ModelView() {};

  void OnAttachHook() override;
  void OnUpdateHook() override;
  void ControlPanelHook() override;
};