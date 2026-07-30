#include "scanning_view.hpp"
#include "camera.hpp"
#include "mesh.hpp"
#include "scene.hpp"
#include <imgui.h>
#include <memory>

void ScanningView::CreateScanningScene() {
  PLOG_DEBUG << "Creating scanning scene...";

  Scene scene;
  scene.SetSceneName("Scanning");

  int cObjectIdx = 0;

  // Non-scene objects
  auto orientationGizmoMesh = CreateOrientationGizmo();
  auto orientationGizmoMaterial = std::make_shared<Material>(MaterialType::OrientationGizmo);
  orientationGizmoMaterial->m_LineWidth = 0.005f;
  auto orientationGizmoObject =
      std::make_shared<Object>(orientationGizmoMesh, orientationGizmoMaterial);
  scene.AddObject(orientationGizmoObject);
  m_GizmosIdx = cObjectIdx++;

  auto reticleMaterial = std::make_shared<Material>(MaterialType::Reticle);
  reticleMaterial->m_LineWidth = 0.005f;

  auto redReticleMesh = CreateReticle(glm::vec3(1.0f, 0.0f, 0.0f));
  auto redReticleObject = std::make_shared<Object>(redReticleMesh, reticleMaterial);
  redReticleObject->m_Active = false;
  scene.AddObject(redReticleObject);
  m_RedReticleIdx = cObjectIdx++;

  auto greenReticleMesh = CreateReticle(glm::vec3(0.0f, 1.0f, 0.0f));
  auto greenReticleObject = std::make_shared<Object>(greenReticleMesh, reticleMaterial);
  greenReticleObject->m_Active = false;
  scene.AddObject(greenReticleObject);
  m_GreenReticleIdx = cObjectIdx++;

  m_MatFlat = std::make_shared<Material>(MaterialType::PointFlat);
  m_MatShaded = std::make_shared<Material>(MaterialType::PointShaded);
  m_MatNormal = std::make_shared<Material>(MaterialType::PointNormal);
  auto placeHolderMesh = std::make_shared<Mesh>();
  auto pointCloudPlaceHolder = std::make_shared<Object>(placeHolderMesh, m_MatShaded);
  scene.AddObject(pointCloudPlaceHolder);
  m_PointCloudIdx = cObjectIdx++;

  // Default camera
  auto perspectiveProjection =
      std::make_shared<PerspectiveProjection>(45.0f, 0.1f, 1000.0f, 16.0f / 9.0f);
  m_InspectionController = std::make_shared<TrackBallController>(
      glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  m_ScanningController = std::make_shared<CameraController>();
  auto camera = std::make_shared<Camera>(perspectiveProjection, m_InspectionController);
  scene.SetCamera(camera);

  m_CurrentScene = std::make_shared<Scene>(scene);

  PLOG_DEBUG << "Scanning scene creation complete";
}

void ScanningView::OnAttachExtra() {
  m_SHMModel.Open("/model", 113246624);
  CreateScanningScene();
  m_VulkanEngine->SetScene(m_CurrentScene);
}

void ScanningView::OnUpdateExtra() {
  if (m_SHMModel.IsOpen()) {
    auto newPCMesh =
        LoadPointCloudFromSharedMemory(m_SHMModel.Data(), m_RevisionNumber, m_Tracking, m_Pose);
    if (newPCMesh) {
      auto newPCObject = std::make_shared<Object>(newPCMesh, m_MatShaded);
      m_CurrentScene->ReplaceObject(m_PointCloudIdx, newPCObject);
      m_VulkanEngine->GetVkScene()->ReplaceObject(m_PointCloudIdx, newPCObject);
    }
  } else {
    m_SHMModel.Open("/model", 113246624);
  }

  if (m_IsScanning && !ImGui::IsAnyMouseDown()) {
    m_CurrentScene->GetCamera()->SetCameraController(m_ScanningController);
    m_ScanningController->SetMatrix(m_Pose);
    if (m_Tracking) {
      m_CurrentScene->GetObjects()[m_GreenReticleIdx]->m_Active = true;
      m_CurrentScene->GetObjects()[m_RedReticleIdx]->m_Active = false;
    } else {
      m_CurrentScene->GetObjects()[m_GreenReticleIdx]->m_Active = false;
      m_CurrentScene->GetObjects()[m_RedReticleIdx]->m_Active = true;
    }
  } else {
    m_CurrentScene->GetCamera()->SetCameraController(m_InspectionController);
  }
}