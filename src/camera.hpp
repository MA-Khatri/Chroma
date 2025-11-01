#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
  Camera(glm::vec3 position, glm::vec3 lookAt, glm::vec3 up, float nearClip = 0.1f,
         float farClip = 1000.0f)
      : m_Position(position), m_LookAt(lookAt), m_Up(up), m_NearClip(nearClip), m_FarClip(farClip) {
    UpdateMatrices();
  }
  ~Camera() = default;

  void SetPosition(const glm::vec3 &position) {
    m_Position = position;
    UpdateMatrices();
  };
  void SetLookAt(const glm::vec3 &lookAt) {
    m_LookAt = lookAt;
    UpdateMatrices();
  };
  void SetUp(const glm::vec3 &up) {
    m_Up = up;
    UpdateMatrices();
  };
  void SetOrientation(const glm::vec3 &position, const glm::vec3 &lookAt, const glm::vec3 &up) {
    m_Position = position;
    m_LookAt = lookAt;
    m_Up = up;
    UpdateMatrices();
  };

  void SetClippingPlanes(float nearClip, float farClip) {
    m_NearClip = nearClip;
    m_FarClip = farClip;
  };
  void SetNearClip(float nearClip) { m_NearClip = nearClip; };
  void SetFarClip(float farClip) { m_FarClip = farClip; };

  const float &GetNearClip() const { return m_NearClip; };
  const float &GetFarClip() const { return m_FarClip; };

  const glm::mat4 &GetViewMatrix() { return m_ViewMatrix; };
  const glm::mat4 &GetProjectionMatrix() { return m_ProjectionMatrix; };
  const glm::mat4 &GetViewProjectionMatrix() { return m_ViewProjectionMatrix; };

protected:
  void UpdateMatrices() {
    m_ViewMatrix = glm::lookAt(m_Position, m_LookAt, m_Up);
    // Note: projectionMatrix should be set in derived classes
    m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
  };

  glm::vec3 m_Position = glm::vec3(0.0f, 0.0f, 5.0f);
  glm::vec3 m_LookAt = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 m_Up = glm::vec3(0.0f, 1.0f, 0.0f);

  glm::mat4 m_ViewMatrix = glm::mat4(1.0f);
  glm::mat4 m_ProjectionMatrix = glm::mat4(1.0f);
  glm::mat4 m_ViewProjectionMatrix = glm::mat4(1.0f);

  float m_NearClip = 0.1f;
  float m_FarClip = 100.0f;
};

class PerspectiveCamera : public Camera {
public:
  PerspectiveCamera(float vfov, float aspectRatio, glm::vec3 position, glm::vec3 lookAt,
                    glm::vec3 up, float nearClip = 0.1f, float farClip = 1000.0f)
      : Camera(position, lookAt, up, nearClip, farClip), m_VFOV(vfov), m_AspectRatio(aspectRatio) {}

  void SetPerspective(float vfov, float aspectRatio) {
    m_VFOV = vfov;
    m_AspectRatio = aspectRatio;
    m_ProjectionMatrix = glm::perspective(glm::radians(vfov), aspectRatio, m_NearClip, m_FarClip);
    UpdateMatrices();
  };

private:
  float m_VFOV = 45.0f; // vertical in degrees
  float m_AspectRatio = 16.0f / 9.0f;
};

class ThinLensCamera : public PerspectiveCamera {
public:
  ThinLensCamera(float aperture, float focalLength, float vfov, float aspectRatio,
                 glm::vec3 position, glm::vec3 lookAt, glm::vec3 up, float nearClip = 0.1f,
                 float farClip = 1000.0f)
      : PerspectiveCamera(vfov, aspectRatio, position, lookAt, up, nearClip, farClip),
        m_Aperture(aperture), m_FocalLength(focalLength) {}

  void SetLensParameters(float aperture, float focalLength) {
    m_Aperture = aperture;
    m_FocalLength = focalLength;
  };

private:
  float m_Aperture = 0.1f;
  float m_FocalLength = 10.0f;
};

class OrthographicCamera : public Camera {
public:
  OrthographicCamera(float left, float right, float bottom, float top, glm::vec3 position,
                     glm::vec3 lookAt, glm::vec3 up, float nearClip = 0.1f, float farClip = 1000.0f)
      : Camera(position, lookAt, up, nearClip, farClip), m_Left(left), m_Right(right),
        m_Bottom(bottom), m_Top(top) {}

  void SetOrthographic(float left, float right, float bottom, float top) {
    m_Left = left;
    m_Right = right;
    m_Bottom = bottom;
    m_Top = top;
    m_ProjectionMatrix = glm::ortho(m_Left, m_Right, m_Bottom, m_Top, m_NearClip, m_FarClip);
    UpdateMatrices();
  };

private:
  float m_Left = -1.0f;
  float m_Right = 1.0f;
  float m_Bottom = -1.0f;
  float m_Top = 1.0f;
};

// TODO: Implement realistic camera