#pragma once

#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <SDL3/SDL.h>
#include <imgui.h>

// Forward declaration of Camera class
class Camera;

// ========================================
// ========== Projection Classes ==========
// ========================================

class Projection {
public:
  virtual ~Projection() = default;
  virtual glm::mat4 GetMatrix() const = 0;

  virtual void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) = 0;

  void SetClippingPlanes(float nearClip, float farClip) {
    m_NearClip = nearClip;
    m_FarClip = farClip;
  }

  void SetNearClip(float nearClip) { m_NearClip = nearClip; }
  void SetFarClip(float farClip) { m_FarClip = farClip; }

  float GetNearClip() const { return m_NearClip; }
  float GetFarClip() const { return m_FarClip; }

  float GetAspectRatio() const { return m_AspectRatio; }
  void SetAspectRatio(float aspectRatio) { m_AspectRatio = aspectRatio; }

protected:
  float m_NearClip = 0.1f;
  float m_FarClip = 1000.0f;
  float m_AspectRatio = 16.0f / 9.0f; // Default aspect ratio
};

class PerspectiveProjection : public Projection {
public:
  PerspectiveProjection(float vfov, float nearClip, float farClip, float aspectRatio) {
    m_VFOV = vfov;
    m_NearClip = nearClip;
    m_FarClip = farClip;
    m_AspectRatio = aspectRatio;
  }

  glm::mat4 GetMatrix() const override {
    return glm::perspective(glm::radians(m_VFOV), m_AspectRatio, m_NearClip, m_FarClip);
  }

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override;

protected:
  float m_VFOV = 45.0f; // vertical field of view in degrees
};

class OrthographicProjection : public Projection {
public:
  OrthographicProjection(float verticalExtent, float nearClip, float farClip, float aspectRatio) {
    m_VerticalExtent = verticalExtent;
    m_NearClip = nearClip;
    m_FarClip = farClip;
    m_AspectRatio = aspectRatio;
  }

  glm::mat4 GetMatrix() const override {
    const float halfVerticalExtent = m_VerticalExtent * 0.5f;
    const float halfHorizontalExtent = halfVerticalExtent * m_AspectRatio;

    return glm::ortho(-halfHorizontalExtent, halfHorizontalExtent, -halfVerticalExtent,
                      halfVerticalExtent, m_NearClip, m_FarClip);
  }

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override;

protected:
  float m_VerticalExtent = 20.0f; // Total visible vertical size in world units
};

// ===============================================
// ========== Camera Controller Classes ==========
// ===============================================

class CameraController {
protected:
  glm::vec3 m_Position = glm::vec3(0.0f, 0.0f, 5.0f);
  glm::vec3 m_LookAt = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 m_Up = glm::vec3(0.0f, 0.0f, 1.0f);

public:
  virtual ~CameraController() = default;

  virtual void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) = 0;

  virtual glm::mat4 GetMatrix() const = 0;

  glm::vec3 GetPosition() const { return m_Position; }
  glm::vec3 GetLookAt() const { return m_LookAt; }
  glm::vec3 GetUp() const { return m_Up; }
};

class FreeFlyController : public CameraController {
public:
  FreeFlyController(glm::vec3 position, float yaw, float pitch) {
    m_Position = position;
    m_Yaw = yaw;
    m_Pitch = pitch;
    m_Up = glm::vec3(0.0f, 0.0f, 1.0f);
    UpdateLookAt();
  }

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override;

  glm::mat4 GetMatrix() const override { return glm::lookAt(m_Position, m_LookAt, m_Up); }

private:
  void UpdateLookAt() {
    glm::vec3 direction;
    direction.x = cos(glm::radians(m_Pitch)) * cos(glm::radians(m_Yaw));
    direction.y = cos(glm::radians(m_Pitch)) * sin(glm::radians(m_Yaw));
    direction.z = sin(glm::radians(m_Pitch));
    m_LookAt = m_Position + glm::normalize(direction);
  }

  float m_Yaw = 0.0f;         // Yaw in degrees from +x axis
  float m_Pitch = 0.0f;       // Pitch in degrees from xy-plane
  float m_Speed = 10.0f;      // Movement speed
  float m_Sensitivity = 0.1f; // Mouse sensitivity

  // Key state for smooth movement
  bool m_MoveForward = false;
  bool m_MoveBackward = false;
  bool m_MoveLeft = false;
  bool m_MoveRight = false;
  bool m_MoveUp = false;
  bool m_MoveDown = false;
};

class OrbitController : public CameraController {
public:
  OrbitController(glm::vec3 center, float radius, float azimuth, float elevation) {
    m_LookAt = center;
    m_Radius = radius;
    m_Azimuth = azimuth;
    m_Elevation = elevation;
    UpdatePosition();
  }

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override;

  glm::mat4 GetMatrix() const override {
    return glm::lookAt(m_Position, m_LookAt, m_Up);
  }

private:
  // Note: Must call this before using m_Position!
  void UpdatePosition() {
    float azimuthRad = glm::radians(m_Azimuth);
    float elevationRad = glm::radians(m_Elevation);

    glm::vec3 offset;
    offset.x = m_Radius * cos(elevationRad) * cos(azimuthRad);
    offset.y = m_Radius * cos(elevationRad) * sin(azimuthRad);
    offset.z = m_Radius * sin(elevationRad);

    m_Position = m_LookAt + offset;
  }

  float m_Radius = 5.0f;    // Distance from the center point
  float m_Azimuth = 0.0f;   // Horizontal angle from +x in degrees
  float m_Elevation = 0.0f; // Vertical angle from xy-plane in degrees
};

class TrackBallController : public CameraController {
public:
  TrackBallController(glm::vec3 center, glm::vec3 position, glm::vec3 up) {
    m_LookAt = center;
    m_Position = position;
    m_Up = glm::normalize(up);
  }

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override;

  glm::mat4 GetMatrix() const override { return glm::lookAt(m_Position, m_LookAt, m_Up); }
};

// ==================================
// ========== Camera Class ==========
// ==================================

class Camera {
public:
  Camera(std::shared_ptr<Projection> projection, std::shared_ptr<CameraController> controller)
      : m_Projection(std::move(projection)), m_Controller(std::move(controller)) {}

  void Update(int64_t deltaTime, const SDL_Event *event = nullptr) {
    if (m_Projection) {
      m_Projection->Update(*this, deltaTime, event);
    }
    if (m_Controller && m_ControllerActive) {
      m_Controller->Update(*this, deltaTime, event);
    }
  }

  glm::mat4 GetViewMatrix() const {
    return m_Controller ? m_Controller->GetMatrix() : glm::mat4(1.0f);
  }

  glm::mat4 GetProjectionMatrix() const {
    return m_Projection ? m_Projection->GetMatrix() : glm::mat4(1.0f);
  }

  glm::mat4 GetViewProjectionMatrix() const { return GetProjectionMatrix() * GetViewMatrix(); }

  glm::vec3 GetPosition() const {
    if (m_Controller) {
      return m_Controller->GetPosition();
    } else
      return glm::vec3(0.0f);
  }

  void SetViewportBounds(ImVec2 min, ImVec2 max) {
    m_ViewportMin = min;
    m_ViewportMax = max;

    int width = max.x - min.x;
    int height = max.y - min.y;

    if (m_Projection) {
      m_Projection->SetAspectRatio(float(width) / float(height));
    }
  }

  ImVec2 GetViewportMin() const { return m_ViewportMin; }
  ImVec2 GetViewportMax() const { return m_ViewportMax; }
  ImVec2 GetViewportSize() const {
    return ImVec2(m_ViewportMax.x - m_ViewportMin.x, m_ViewportMax.y - m_ViewportMin.y);
  }

  void SetControllerActive(bool active) { m_ControllerActive = active; }
  bool IsControllerActive() const { return m_ControllerActive; }

private:
  std::shared_ptr<Projection> m_Projection;
  std::shared_ptr<CameraController> m_Controller;

  bool m_ControllerActive = false;

  ImVec2 m_ViewportMin;
  ImVec2 m_ViewportMax;
};