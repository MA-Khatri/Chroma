#pragma once

#include "camera.hpp"
#include "object.hpp"
#include <filesystem>
#include <memory>

class Scene {

public:
  Scene() {};
  Scene(std::filesystem::path filePath); // Load scene from file (TODO)
  ~Scene() = default;

  void SetCamera(const std::shared_ptr<Camera> cam) { m_Camera = cam; }
  std::shared_ptr<Camera> GetCamera() const { return m_Camera; }
  
  void SetSceneName(const std::string &name) { m_SceneName = name; }
  const std::string &GetSceneName() const { return m_SceneName; }

  void AddObject(const Object &object) { m_Objects.push_back(object); }
  
  private:
  std::string m_SceneName = "Untitled Scene";
  std::shared_ptr<Camera> m_Camera;
  std::vector<Object> m_Objects;
};

Scene CreateTestScene();