#pragma once

#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <SDL3/SDL.h>

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

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override {
    // TODO: Update vfov, clipping planes, etc. based on user input
  }

protected:
  float m_VFOV = 45.0f; // vertical field of view in degrees
};

class OrthographicProjection : public Projection {
public:
  OrthographicProjection(float left, float right, float bottom, float top, float nearClip,
                         float farClip, float aspectRatio) {
    m_Left = left;
    m_Right = right;
    m_Bottom = bottom;
    m_Top = top;
    m_NearClip = nearClip;
    m_FarClip = farClip;
    m_AspectRatio = aspectRatio;
  }

  glm::mat4 GetMatrix() const override {
    return glm::ortho(m_Left * m_AspectRatio, m_Right * m_AspectRatio, m_Bottom, m_Top, m_NearClip,
                      m_FarClip);
  }

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override {
    // TODO: Update orthographic projection parameters based on user input
  }

protected:
  float m_Left = -10.0f;
  float m_Right = 10.0f;
  float m_Bottom = -10.0f;
  float m_Top = 10.0f;
};

// ===============================================
// ========== Camera Controller Classes ==========
// ===============================================

class CameraController {
public:
  virtual ~CameraController() = default;

  virtual void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) = 0;

  virtual glm::mat4 GetMatrix() const = 0;

protected:
  glm::vec3 m_Position = glm::vec3(0.0f, 0.0f, 5.0f);
  glm::vec3 m_LookAt = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
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

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override {
    // Process incoming event (if any) to update state, but always
    // apply movement each frame based on held keys for smooth motion.
    if (event) {
      switch (event->type) {
      case SDL_EVENT_MOUSE_MOTION: {
        if (event->motion.state &
            SDL_BUTTON_LMASK) { // Only rotate when left mouse button is pressed
          m_Yaw -= event->motion.xrel * m_Sensitivity;
          m_Pitch -= event->motion.yrel * m_Sensitivity;

          // Clamp pitch to avoid gimbal lock
          if (m_Pitch > 89.0f)
            m_Pitch = 89.0f;
          if (m_Pitch < -89.0f)
            m_Pitch = -89.0f;

          UpdateLookAt();
        }
        break;
      }
      case SDL_EVENT_KEY_DOWN:
      case SDL_EVENT_KEY_UP: {
        bool pressed = (event->type == SDL_EVENT_KEY_DOWN);
        // Track key states instead of moving only on key-down events
        switch (event->key.key) {
        case SDLK_W:
          m_MoveForward = pressed;
          break;
        case SDLK_S:
          m_MoveBackward = pressed;
          break;
        case SDLK_A:
          m_MoveLeft = pressed;
          break;
        case SDLK_D:
          m_MoveRight = pressed;
          break;
        case SDLK_Q:
        case SDLK_SPACE:
          m_MoveUp = pressed;
          break;
        case SDLK_E:
        case SDLK_LSHIFT:
          m_MoveDown = pressed;
          break;
        default:
          break;
        }
        break;
      }
      default:
        break;
      }
    }

    // Apply continuous movement based on held keys
    float velocity = m_Speed * deltaTime / 1e9f; // Convert nanoseconds to seconds

    glm::vec3 view_dir = glm::normalize(m_LookAt - m_Position);
    glm::vec3 right_dir = glm::normalize(glm::cross(view_dir, m_Up));

    glm::vec3 delta = glm::vec3(0.0f);
    if (m_MoveForward)
      delta += velocity * view_dir;
    if (m_MoveBackward)
      delta -= velocity * view_dir;
    if (m_MoveLeft)
      delta -= velocity * right_dir;
    if (m_MoveRight)
      delta += velocity * right_dir;
    if (m_MoveUp)
      delta += velocity * m_Up;
    if (m_MoveDown)
      delta -= velocity * m_Up;

    if (glm::length(delta) > 0.0f) {
      m_Position += delta;
      UpdateLookAt();
    }
  }

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
  OrbitController(glm::vec3 center, float radius, float azimuth, float elevation)
      : m_Center(center), m_Radius(radius), m_Azimuth(azimuth), m_Elevation(elevation) {}

  void Update(Camera &camera, int64_t deltaTime, const SDL_Event *event = nullptr) override {
    // TODO: Implement orbit camera controls (e.g., rotate around a target point, zoom in/out)
  }

  glm::mat4 GetMatrix() const override {
    float azimuthRad = glm::radians(m_Azimuth);
    float elevationRad = glm::radians(m_Elevation);

    glm::vec3 offset;
    offset.x = m_Radius * cos(elevationRad) * cos(azimuthRad);
    offset.y = m_Radius * cos(elevationRad) * sin(azimuthRad);
    offset.z = m_Radius * sin(elevationRad);

    glm::vec3 position = m_Center + offset;
    return glm::lookAt(position, m_Center, m_Up);
  }

private:
  glm::vec3 m_Center = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 m_Up = glm::vec3(0.0f, 0.0f, 1.0f);
  float m_Radius = 5.0f;    // Distance from the center point
  float m_Azimuth = 0.0f;   // Horizontal angle from +x in degrees
  float m_Elevation = 0.0f; // Vertical angle from xy-plane in degrees
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

  void SetAspectRatio(float aspectRatio) {
    if (m_Projection) {
      m_Projection->SetAspectRatio(aspectRatio);
    }
  }

  void SetControllerActive(bool active) { m_ControllerActive = active; }
  bool IsControllerActive() const { return m_ControllerActive; }

private:
  std::shared_ptr<Projection> m_Projection;
  std::shared_ptr<CameraController> m_Controller;

  bool m_ControllerActive = false;
};