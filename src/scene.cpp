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

  // TODO: Add ground grid

  auto planeMesh = std::make_shared<Mesh>(CreatePlaneMesh(10.0f, 10.0f, 10, 10));
  TexturePaths planeTextures;
  planeTextures.albedo = "C:/Users/mmrsk/Repos/Chroma/res/textures/texture.jpg";
  auto planeMaterial = std::make_shared<Material>(planeTextures, MaterialType::Lambertian);
  auto planeTransform = std::make_shared<Transform>();
  auto planeObject = std::make_shared<Object>(planeMesh, planeMaterial, planeTransform);
  scene.AddObject(planeObject);

  auto cubeMesh = std::make_shared<Mesh>(CreateCubeMesh());
  TexturePaths cubeTextures;
  cubeTextures.albedo = "C:/Users/mmrsk/Repos/Chroma/res/textures/texture.jpg";
  auto cubeMaterial = std::make_shared<Material>(cubeTextures, MaterialType::Lambertian);
  auto cubeTransform = std::make_shared<Transform>();
  cubeTransform->SetPosition(
      glm::vec3(0.0f, 0.0f, 0.5f + 1e-4f)); // Position the cube above the plane
  auto cubeObject = std::make_shared<Object>(cubeMesh, cubeMaterial, cubeTransform);
  scene.AddObject(cubeObject);

  auto perspectiveProjection =
      std::make_shared<PerspectiveProjection>(45.0f, 0.1f, 1000.0f, 16.0f / 9.0f);
  auto freeFlyController = std::make_shared<FreeFlyController>(
      glm::vec3(5.0f, 5.0f, 5.0f), -135.0f, -30.0f); // Position and orientation to look at the cube
  auto camera = std::make_shared<Camera>(perspectiveProjection, freeFlyController);
  scene.SetCamera(camera);

  return scene;
}