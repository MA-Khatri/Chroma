#pragma once

#include "camera.hpp"
#include "raster_view.hpp"
#include "sharing/cross_platform_shared_memory.hpp"

class ScanningView : public RasterView {
public:
  ScanningView(std::string name) : RasterView(name) {};
  ~ScanningView() {};

  void OnAttachExtra() override;
  void OnUpdateExtra() override;
  void ControlPanelExtra() override;

private:
  void CreateScanningScene();

  std::shared_ptr<TrackBallController> m_InspectionController;
  std::shared_ptr<ScannerController> m_ScanningController;

  int m_GizmosIdx = -1;
  int m_RedReticleIdx = -1;
  int m_GreenReticleIdx = -1;
  int m_PointCloudIdx = -1;

  std::shared_ptr<Material> m_MatFlat;
  std::shared_ptr<Material> m_MatShaded;
  std::shared_ptr<Material> m_MatNormal;

  CrossPlatformSharedMemory m_SHMModel;
  CrossPlatformSharedMemory m_SHMGizmos;

  uint32_t m_RevisionNumber;
  uint32_t m_Tracking;

  glm::mat4 m_ScannerPose;
  glm::mat4 m_ScannerView;

  bool m_UseScannerPose = false;
};