#pragma once

#include "camera.hpp"
#include "object.hpp"
#include <filesystem>
#include <memory>

static int SceneCounter = 0;

class Scene {
public:
  Scene() {};
  Scene(std::filesystem::path filePath); // Load scene from file (TODO)
  ~Scene() = default;

  void SetCamera(const std::shared_ptr<Camera> cam) { m_Camera = cam; }
  std::shared_ptr<Camera> GetCamera() const { return m_Camera; }

  void SetSceneName(const std::string &name) { m_SceneName = name; }
  const std::string &GetSceneName() const { return m_SceneName; }

  void AddObject(std::shared_ptr<Object> object) { m_Objects.push_back(object); }
  const std::vector<std::shared_ptr<Object>> &GetObjects() const { return m_Objects; }

  void SetClearColor(const glm::vec3 &color) { m_ClearColor = color; }
  const glm::vec3 &GetClearColor() const { return m_ClearColor; }

  const int m_SceneID = SceneCounter++;

private:
  std::string m_SceneName = "Untitled Scene";
  std::shared_ptr<Camera> m_Camera;
  std::vector<std::shared_ptr<Object>> m_Objects;
  glm::vec3 m_ClearColor = glm::vec3(0.901f, 0.913f, 0.960f); // off-white
  // glm::vec3 m_ClearColor = glm::vec3(0.2470588f, 0.2470588f, 0.2470588f);
};

Scene CreateTestScene();

Scene CreateScanningScene();