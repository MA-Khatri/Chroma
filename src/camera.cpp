#include "camera.hpp"

#include <plog/Log.h>

// ===================================
// === Event-based Update Handlers ===
// ===================================

//
// Projection update handlers
//
void PerspectiveProjection::Update(Camera &camera, int64_t deltaTime, const SDL_Event *event) {
  if (event) {
    switch (event->type) {
    case SDL_EVENT_MOUSE_WHEEL: {
      const SDL_Keymod modState = SDL_GetModState();

      if (modState & SDL_KMOD_CTRL) { // Ctrl + mouse wheel for far clip adjustment
        m_FarClip += event->wheel.y;
        // Ensure far clip is always greater than near clip
        if (m_FarClip < m_NearClip + 0.1f)
          m_FarClip = m_NearClip + 0.1f;

        PLOG_DEBUG << "Adjusted far clip to: " << m_FarClip;
      }

      else if (modState & SDL_KMOD_ALT) { // Alt + mouse wheel for near clip adjustment
        m_NearClip += event->wheel.y;
        // Ensure near clip is always less than far clip
        if (m_NearClip > m_FarClip - 0.1f)
          m_NearClip = m_FarClip - 0.1f;

        PLOG_DEBUG << "Adjusted near clip to: " << m_NearClip;
      }

      else { // Zoom in/out by adjusting the vertical field of view (vfov)
        m_VFOV -= event->wheel.y;
        // Clamp vfov to a reasonable range (e.g., 1 to 120 degrees)
        if (m_VFOV < 1.0f)
          m_VFOV = 1.0f;
        if (m_VFOV > 120.0f)
          m_VFOV = 120.0f;

        PLOG_DEBUG << "Adjusted vertical field of view (vfov) to: " << m_VFOV;
      }
      break;
    }

    default:
      break;
    }
  }
}

void OrthographicProjection::Update(Camera &camera, int64_t deltaTime, const SDL_Event *event) {
  // TODO: Update orthographic projection parameters based on user input
}

//
// Camera controller update handlers
//
void FreeFlyController::Update(Camera &camera, int64_t deltaTime, const SDL_Event *event) {
  // Process incoming event (if any) to update state, but always
  // apply movement each frame based on held keys for smooth motion.
  if (event) {
    switch (event->type) {
    case SDL_EVENT_MOUSE_MOTION: {
      if (event->motion.state & SDL_BUTTON_LMASK) { // Only rotate when left mouse button is pressed
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

void OrbitController::Update(Camera &camera, int64_t deltaTime, const SDL_Event *event) {
  // TODO: Implement orbit camera controls (e.g., rotate around a target point, zoom in/out)
  // Make sure to update m_Position to be the resulting camera position (distinct from the orbit
  // center!)
}