#pragma once

#include "camera.hpp"
#include "object.hpp"
#include <memory>

class Scene {

public:
  Scene() {};
  Scene(std::string filePath); // Load scene from file (TODO)
  ~Scene() = default;

  void AddObject(const Object &object) { objects.push_back(object); }
  void SetCamera(const std::shared_ptr<Camera> cam) { camera = cam; }

private:
  std::shared_ptr<Camera> camera;
  std::vector<Object> objects;
};

Scene CreateTestScene();