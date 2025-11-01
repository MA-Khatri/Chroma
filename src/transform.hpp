#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "glm/gtx/quaternion.hpp"

class Transform {
public:
  Transform() = default;
  Transform(const glm::vec3 &position, const glm::vec3 &rotation, const glm::vec3 &scale)
      : m_Position(position), m_Rotation(rotation), m_Scale(scale) {}
  ~Transform() = default;

  void SetPosition(const glm::vec3 &position) { m_Position = position; };
  void SetRotation(const glm::vec3 &rotation) { m_Rotation = rotation; };
  void SetScale(const glm::vec3 &scale) { m_Scale = scale; };

private:
  glm::vec3 m_Position = glm::vec3(0.0f);
  glm::quat m_Rotation = glm::quat(glm::vec3(0.0f));
  glm::vec3 m_Scale = glm::vec3(1.0f);

  glm::mat4 m_ModelMatrix = glm::mat4(1.0f);

  glm::mat4 UpdateMatrix() const {
    glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), m_Position);
    glm::mat4 rotationMat = glm::toMat4(m_Rotation);
    glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), m_Scale);

    return translationMat * rotationMat * scaleMat;
  }
};