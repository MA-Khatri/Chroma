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

  auto clearColor = glm::vec3(0.2470588f, 0.2470588f, 0.2470588f); // Dark gray
  scene.SetClearColor(clearColor);

  // Note: Store the clear color in the normal matrix for the ground grid shader
  auto groundGridMesh = std::make_shared<Mesh>(CreateGroundGridMesh());
  auto groundGridMaterial = std::make_shared<Material>(MaterialType::GroundGrid);
  auto groundGridTransform = std::make_shared<Transform>();
  groundGridTransform->SetNormalMatrix(glm::mat3(clearColor, clearColor, clearColor));
  auto groundGridObject =
      std::make_shared<Object>(groundGridMesh, groundGridMaterial, groundGridTransform);
  scene.AddObject(groundGridObject);

  auto axesMesh = std::make_shared<Mesh>(CreateXYAxesMesh());
  auto axesMaterial = std::make_shared<Material>(MaterialType::GroundGrid);
  axesMaterial->m_LineWidth = 2.0f; // Set thicker line width for axes
  auto axesTransform = std::make_shared<Transform>();
  axesTransform->SetNormalMatrix(glm::mat3(clearColor, clearColor, clearColor));
  auto axesObject = std::make_shared<Object>(axesMesh, axesMaterial, axesTransform);
  scene.AddObject(axesObject);

  auto planeMesh = std::make_shared<Mesh>(CreatePlaneMesh(10.0f, 10.0f, 10, 10));
  TexturePaths planeTextures;
  planeTextures.albedo = "checker.png";
  auto planeMaterial = std::make_shared<Material>(planeTextures, MaterialType::Conductor);
  auto planeTransform = std::make_shared<Transform>();
  planeTransform->SetPosition(glm::vec3(0.0f, 0.0f, -1e-2f));
  auto planeObject = std::make_shared<Object>(planeMesh, planeMaterial, planeTransform);
  scene.AddObject(planeObject);

  auto cubeMesh = std::make_shared<Mesh>(CreateCubeMesh());
  TexturePaths cubeTextures;
  cubeTextures.albedo = "texture.jpg";
  auto cubeMaterial = std::make_shared<Material>(cubeTextures, MaterialType::Lambertian);
  auto cubeTransform = std::make_shared<Transform>();
  cubeTransform->SetPosition(
      glm::vec3(0.0f, 0.0f, 0.5f + 1e-4f)); // Position the cube above the plane
  auto cubeObject = std::make_shared<Object>(cubeMesh, cubeMaterial, cubeTransform);
  scene.AddObject(cubeObject);

  auto bunnyMesh = std::make_shared<Mesh>(LoadMeshFromFile("bunny.obj"));
  auto bunnyMaterial = std::make_shared<Material>(MaterialType::Lambertian);
  auto bunnyTransform = std::make_shared<Transform>();
  bunnyTransform->SetPosition(
      glm::vec3(1.0f, 0.0f, 0.0f)); // Position the bunny to the right of the cube
  auto bunnyObject = std::make_shared<Object>(bunnyMesh, bunnyMaterial, bunnyTransform);
  scene.AddObject(bunnyObject);

  auto pointsMesh = std::make_shared<Mesh>(LoadMeshFromFile("thumb_print.ply"));
  auto pointsMaterial = std::make_shared<Material>(MaterialType::PointNormal);
  auto pointsTransform = std::make_shared<Transform>();
  pointsTransform->SetRotation(glm::vec3(3.14159265f, 0.0f, 0.0f));
  pointsTransform->SetPosition(glm::vec3(0.0f, 0.0f, 100.0f));
  auto pointsObject = std::make_shared<Object>(pointsMesh, pointsMaterial, pointsTransform);
  scene.AddObject(pointsObject);

  auto perspectiveProjection =
      std::make_shared<PerspectiveProjection>(45.0f, 0.1f, 1000.0f, 16.0f / 9.0f);
  auto freeFlyController = std::make_shared<FreeFlyController>(
      glm::vec3(5.0f, 5.0f, 5.0f), -135.0f, -30.0f); // Position and orientation to look at the cube
  auto camera = std::make_shared<Camera>(perspectiveProjection, freeFlyController);
  scene.SetCamera(camera);

  return scene;
}