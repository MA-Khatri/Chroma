#include "model_view.hpp"
#include "scene.hpp"

void ModelView::OnAttachHook() {
  m_CurrentScene = std::make_shared<Scene>(CreateTestScene());
  // TODO
}

void ModelView::OnUpdateHook() {
  // TODO
}

void ModelView::ControlPanelHook() {
  ImGui::SeparatorText("Scanning");

  auto sceneObjects = m_CurrentScene->GetObjects();
  int objectCount = 0;
  const int nNonSceneObjects = 1; // TODO: set this as a class member when creating scene
  for (auto &object : sceneObjects) {
    if (objectCount < nNonSceneObjects) {
      // Skip non-scene objects, e.g. orientation gizmo
      objectCount++;
      continue;
    }
    int vertexCount = object->m_Mesh->vertices.size();
    ImGui::Text("Object %i Vertex Count: %i", objectCount++ - nNonSceneObjects, vertexCount);
  }
}