#include "scene.hpp"

Scene::Scene(std::string filePath) {
  // TODO: Load scene from file
}

Scene CreateTestScene() {
  Scene scene;

  // Create a simple test scene with a cube and a camera
  auto cubeMesh = std::make_shared<Mesh>(CreateCubeMesh());
  auto cubeMaterial = std::make_shared<Material>();
  Transform cubeTransform;
  Object cubeObject(cubeMesh, cubeMaterial, cubeTransform);
  scene.AddObject(cubeObject);

  auto camera = std::make_shared<Camera>();
  scene.SetCamera(camera);

  return scene;
}