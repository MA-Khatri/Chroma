#pragma once

#include <glm/glm.hpp>

// TODO: Add camera types (e.g., orthographic)

class Camera {
public:
  Camera(float vfov, float aspectRatio, float nearClip, float farClip, glm::vec3 position,
         glm::vec3 lookAt, glm::vec3 up)
      : vfov(vfov), aspectRatio(aspectRatio), nearClip(nearClip), farClip(farClip),
        position(position), lookAt(lookAt), up(up) {
    UpdateMatrices();
  }
  Camera();
  ~Camera() = default;

  void SetPosition(const glm::vec3 &pos);
  void SetLookAt(const glm::vec3 &target);
  void SetUp(const glm::vec3 &upVector);
  void SetPerspective(float vfov, float aspectRatio, float nearClip, float farClip);

  const glm::mat4 &GetViewMatrix();
  const glm::mat4 &GetProjectionMatrix();
  const glm::mat4 &GetViewProjectionMatrix();

private:
  void UpdateMatrices();

  glm::vec3 position = glm::vec3(0.0f, 0.0f, 5.0f);
  glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::mat4 viewMatrix;
  glm::mat4 projectionMatrix;
  glm::mat4 viewProjectionMatrix;

  float vfov = 45.0f; // vertical in degrees
  float aspectRatio = 16.0f / 9.0f;
  float nearClip = 0.1f;
  float farClip = 100.0f;
};