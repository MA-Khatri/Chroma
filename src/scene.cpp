#include "scene.hpp"
#include "camera.hpp"
#include "texture.hpp"
#include <memory>

Scene::Scene(std::filesystem::path filePath) {
  // TODO: Load scene from file
}

Scene CreateTestScene() {
  // Create a simple test scene with a few objects and a camera
  Scene scene;
  scene.SetSceneName("Test Scene");

  // Scene objects
  auto pointsMesh = std::make_shared<Mesh>(LoadMeshFromFile("multimaterial.ply"));
  auto pointsMaterial = std::make_shared<Material>(MaterialType::PointShaded);
  auto pointsTransform = std::make_shared<Transform>();
  pointsTransform->SetRotation(glm::vec3(180.0f, 0.0f, 0.0f));
  pointsTransform->SetPosition(glm::vec3(0.0f, 0.0f, 100.0f));
  auto pointsObject = std::make_shared<Object>(pointsMesh, pointsMaterial, pointsTransform);
  scene.AddObject(pointsObject);

  // Default camera
  auto perspectiveProjection =
      std::make_shared<PerspectiveProjection>(45.0f, 0.1f, 1000.0f, 16.0f / 9.0f);
  auto trackBallController = std::make_shared<TrackBallController>(
      glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  auto camera = std::make_shared<Camera>(perspectiveProjection, trackBallController);
  scene.SetCamera(camera);

  return scene;
}