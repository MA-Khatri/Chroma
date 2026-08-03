#include "scanning_view.hpp"

#include "camera.hpp"
#include "mesh.hpp"
#include "scene.hpp"

#include <glm/gtc/matrix_inverse.hpp>
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
  pointCloudPlaceHolder->m_Active = false;
  scene.AddObject(pointCloudPlaceHolder);
  m_PointCloudIdx = cObjectIdx++;

  // Default camera
  auto perspectiveProjection =
      std::make_shared<PerspectiveProjection>(45.0f, 0.1f, 1000.0f, 16.0f / 9.0f);
  m_InspectionController = std::make_shared<TrackBallController>(
      glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  m_ScanningController = std::make_shared<ScannerController>();
  auto camera = std::make_shared<Camera>(perspectiveProjection, m_InspectionController);
  scene.SetCamera(camera);

  m_CurrentScene = std::make_shared<Scene>(scene);

  PLOG_DEBUG << "Scanning scene creation complete";
}

void ScanningView::OnAttachHook() {
  m_SHMModel.Open("/model", 113246624);
  CreateScanningScene();
}

void ScanningView::OnUpdateHook() {
  if (m_SHMModel.IsOpen()) {
    auto newPCMesh = LoadPointCloudFromSharedMemory(m_SHMModel.Data(), m_RevisionNumber, m_Tracking,
                                                    m_ScannerPose);
    if (newPCMesh) {
      auto newPCObject = std::make_shared<Object>(newPCMesh, m_MatShaded);
      m_VulkanEngine->GetVkScene()->ReplaceObject(m_PointCloudIdx, newPCObject);

      // Apply transformation to make teeth appear upright while scanning and transpose x, y, to
      // account for the transpose we do before sending data to the digitizer
      constexpr glm::mat4 t1 = glm::mat4(0.0f, -1.0f, 0.0f, 0.0f, // c1
                                         -1.0f, 0.0f, 0.0f, 0.0f, // c2
                                         0.0f, 0.0f, -1.0f, 0.0f, // c3
                                         0.0f, 0.0f, 0.0f, 1.0f); // c4

      // TODO: Pass these into reticle vertex shader
      constexpr float reticleVFoV = 9.5f;
      constexpr float reticleAspect = 0.75f;

      // Adjust view frustum to center reticle over scan region
      constexpr glm::mat4 t2 =
          glm::translate(glm::mat4(1.0f), glm::vec3(-reticleVFoV / 2.0f - 1.0f, 0.5f, 0.0f));

      m_ScannerView = t2 * t1 * glm::inverseTranspose(m_ScannerPose);
    }
  } else {
    m_SHMModel.Open("/model", 113246624);
  }

  if (m_UseScannerPose &&
      !(ImGui::IsAnyMouseDown() && m_AppHandle->m_FocusedWindow == m_WindowID)) {
    m_ScanningController->SetMatrix(m_ScannerView);
    m_CurrentScene->GetCamera()->SetCameraController(m_ScanningController);
    if (m_Tracking) {
      m_CurrentScene->GetObjects()[m_GreenReticleIdx]->m_Active = true;
      m_CurrentScene->GetObjects()[m_RedReticleIdx]->m_Active = false;
    } else {
      m_CurrentScene->GetObjects()[m_GreenReticleIdx]->m_Active = false;
      m_CurrentScene->GetObjects()[m_RedReticleIdx]->m_Active = true;
    }
  } else {
    m_CurrentScene->GetObjects()[m_GreenReticleIdx]->m_Active = false;
    m_CurrentScene->GetObjects()[m_RedReticleIdx]->m_Active = false;
    m_CurrentScene->GetCamera()->SetCameraController(m_InspectionController);
  }
}

void ScanningView::ControlPanelHook() {
  ImGui::SeparatorText("Scanning");

  ImGui::Checkbox("Use Scanner Pose", &m_UseScannerPose);

  auto pc = m_CurrentScene->GetObjects()[m_PointCloudIdx];
  if (pc->m_Active) {
    int vertexCount = pc->m_Mesh->vertices.size();
    ImGui::Text("Vertex Count: %i", vertexCount);
  }

  // TODO
}