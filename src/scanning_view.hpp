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

private:
  void CreateScanningScene();

  std::shared_ptr<CameraController> m_InspectionController;
  std::shared_ptr<CameraController> m_ScanningController;

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
  glm::mat4 m_Pose;

  bool m_IsScanning = false;
};