#include "scene.hpp"
#include "camera.hpp"
#include "texture.hpp"
#include <memory>

Scene::Scene(std::filesystem::path filePath) {
  // TODO: Load scene from file
}

Scene CreateTestScene() {
  Scene scene;
  scene.SetSceneName("Test Scene");

  // Create a simple test scene with a cube and a camera
  auto cubeMesh = std::make_shared<Mesh>(CreateCubeMesh());
  TexturePaths cubeTextures;
  cubeTextures.albedo = "C:/Users/mmrsk/Repos/Chroma/res/textures/texture.jpg";
  auto cubeMaterial = std::make_shared<Material>(cubeTextures, MaterialType::Lambertian);
  auto cubeTransform = std::make_shared<Transform>();
  auto cubeObject = std::make_shared<Object>(cubeMesh, cubeMaterial, cubeTransform);
  scene.AddObject(cubeObject);

  auto perspectiveProjection =
      std::make_shared<PerspectiveProjection>(45.0f, 0.1f, 1000.0f, 16.0f / 9.0f);
  auto freeFlyController = std::make_shared<FreeFlyController>(
      glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  auto camera = std::make_shared<Camera>(perspectiveProjection, freeFlyController);
  scene.SetCamera(camera);

  return scene;
}