#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include "glm/gtx/quaternion.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Transform {
public:
  Transform() = default;
  Transform(const glm::vec3 &position, const glm::vec3 &rotation, const glm::vec3 &scale)
      : m_Position(position), m_Rotation(rotation), m_Scale(scale) {}
  ~Transform() = default;

  void SetPosition(const glm::vec3 &position) {
    m_Position = position;
    UpdateMatrix();
  };
  void SetPosition(float x, float y, float z) { SetPosition(glm::vec3(x, y, z)); }

  void SetRotation(const glm::vec3 &rotation) {
    m_Rotation = glm::radians(rotation);
    UpdateMatrix();
  };
  void SetRotation(float x, float y, float z) { SetRotation(glm::vec3(x, y, z)); }

  void SetScale(const glm::vec3 &scale) {
    m_Scale = scale;
    UpdateMatrix();
  };
  void SetScale(float x, float y, float z) { SetScale(glm::vec3(x, y, z)); }
  void SetScale(float scale) { SetScale(glm::vec3(scale)); }

  void Translate(const glm::vec3 &delta) {
    m_Position += delta;
    UpdateMatrix();
  }
  void Translate(float x, float y, float z) { Translate(glm::vec3(x, y, z)); }

  void Rotate(const glm::vec3 &delta) {
    m_Rotation *= glm::quat(glm::radians(delta));
    UpdateMatrix();
  };
  void Rotate(float x, float y, float z) { Rotate(glm::vec3(x, y, z)); }

  void Scale(const glm::vec3 &factor) {
    m_Scale *= factor;
    UpdateMatrix();
  };
  void Scale(float x, float y, float z) { Scale(glm::vec3(x, y, z)); }
  void Scale(float factor) { Scale(glm::vec3(factor)); }

  void SetModelMatrix(const glm::mat4 &modelMatrix) {
    m_ModelMatrix = modelMatrix;
    m_NormalMatrix = glm::mat3(glm::transpose(glm::inverse(m_ModelMatrix)));
  }

  void SetNormalMatrix(const glm::mat4 &normalMatrix) { m_NormalMatrix = glm::mat3(normalMatrix); }

  void SetNormalMatrix(const glm::mat3 &normalMatrix) { m_NormalMatrix = normalMatrix; }

  const glm::vec3 &GetPosition() { return m_Position; }
  const glm::quat &GetRotation() { return m_Rotation; }
  const glm::vec3 &GetScale() { return m_Scale; }

  const glm::mat4 &GetModelMatrix() { return m_ModelMatrix; }
  const glm::mat3 GetNormalMatrix() { return m_NormalMatrix; }

private:
  glm::vec3 m_Position = glm::vec3(0.0f);
  glm::quat m_Rotation = glm::quat(glm::vec3(0.0f));
  glm::vec3 m_Scale = glm::vec3(1.0f);

  glm::mat4 m_ModelMatrix = glm::mat4(1.0f);
  glm::mat3 m_NormalMatrix = glm::mat3(1.0f);

  void UpdateMatrix() {
    // TODO: is there a glm function that does this directly?
    glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), m_Position);
    glm::mat4 rotationMat = glm::toMat4(m_Rotation);
    glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), m_Scale);

    m_ModelMatrix = translationMat * rotationMat * scaleMat;
    m_NormalMatrix = glm::mat3(glm::transpose(glm::inverse(m_ModelMatrix)));
  }
};