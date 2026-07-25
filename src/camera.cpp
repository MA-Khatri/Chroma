#include "camera.hpp"

#include <SDL3/SDL_events.h>
#include <glm/ext/vector_float2.hpp>
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

      // Ctrl + mouse wheel for far clip adjustment
      if (modState & SDL_KMOD_CTRL) {
        m_FarClip += event->wheel.y;
        if (m_FarClip < m_NearClip + 1.0f)
          m_FarClip = m_NearClip + 1.0f;

        PLOG_DEBUG << "Adjusted far clip to: " << m_FarClip;
      }

      // Alt + mouse wheel for near clip adjustment
      else if (modState & SDL_KMOD_ALT) {
        m_NearClip += event->wheel.y;
        if (m_NearClip > m_FarClip - 1.0f)
          m_NearClip = m_FarClip - 1.0f;

        PLOG_DEBUG << "Adjusted near clip to: " << m_NearClip;
      }

      // Shift + mouse wheel for FOV (zoom in/out)
      else if (modState & SDL_KMOD_SHIFT) {
        m_VFOV -= event->wheel.y;
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
  if (event) {
    switch (event->type) {
    case SDL_EVENT_MOUSE_WHEEL: {
      const SDL_Keymod modState = SDL_GetModState();

      // Ctrl + mouse wheel for far clip adjustment
      if (modState & SDL_KMOD_CTRL) {
        m_FarClip += event->wheel.y;
        if (m_FarClip < m_NearClip + 1.0f)
          m_FarClip = m_NearClip + 1.0f;

        PLOG_DEBUG << "Adjusted far clip to: " << m_FarClip;
      }

      // Alt + mouse wheel for near clip adjustment
      else if (modState & SDL_KMOD_ALT) {
        m_NearClip += event->wheel.y;
        if (m_NearClip > m_FarClip - 1.0f)
          m_NearClip = m_FarClip - 1.0f;

        PLOG_DEBUG << "Adjusted near clip to: " << m_NearClip;
      }

      // Shift + mouse wheel for orthographic extent
      else if (modState & SDL_KMOD_SHIFT) {
        m_VerticalExtent -= event->wheel.y;
        if (m_VerticalExtent < 1.0f)
          m_VerticalExtent = 1.0f;

        PLOG_DEBUG << "Adjusted orthographic vertical extent to: " << m_VerticalExtent;
      }
      break;
    }

    default:
      break;
    }
  }
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
      // Only rotate when left mouse button is pressed
      if (event->motion.state & SDL_BUTTON_LMASK) {
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
  if (event) {
    const SDL_Keymod modState = SDL_GetModState();

    switch (event->type) {
    case SDL_EVENT_MOUSE_MOTION: {
      const uint32_t buttonState = event->motion.state;
      float dx = static_cast<float>(event->motion.xrel);
      float dy = static_cast<float>(event->motion.yrel);

      UpdatePosition();
      glm::vec3 toCenter = m_LookAt - m_Position;
      glm::vec3 right = glm::normalize(glm::cross(toCenter, m_Up));
      glm::vec3 up = glm::normalize(glm::cross(right, toCenter));

      // Pan with right click and drag or shift + left click and drag
      if ((buttonState & SDL_BUTTON_RMASK) ||
          ((modState & SDL_KMOD_SHIFT) && (buttonState & SDL_BUTTON_LMASK))) {
        const float panScale = 0.001f;
        float dist = panScale * glm::length(toCenter);
        dx *= dist;
        dy *= dist;

        glm::vec3 pan = right * dx - up * dy;
        m_LookAt += pan;
        UpdatePosition();
        break;
      }

      // Rotate with left click and drag
      if (buttonState & SDL_BUTTON_LMASK) {
        const float rotateScale = 0.1f;
        m_Azimuth -= dx * rotateScale;
        m_Elevation += dy * rotateScale;
        if (m_Elevation > 89.0f)
          m_Elevation = 89.0f;
        if (m_Elevation < -89.0f)
          m_Elevation = -89.0f;
        break;
      }
      break;
    }

    // Change orbit radius with mouse scroll
    case SDL_EVENT_MOUSE_WHEEL: {
      if (modState & SDL_KMOD_SHIFT) {
        // Shift + mouse wheel is reserved for changing fov/orthographic extent
        break;
      }

      const float minRadius = 0.15f;
      const float radiusScale = 1.1f;
      const float invRadiusScale = 1.0f / radiusScale;
      float delta = event->wheel.y;
      if (delta > 0.0f) {
        m_Radius *= invRadiusScale;
      } else {
        m_Radius *= radiusScale;
      }
      if (m_Radius < minRadius)
        m_Radius = minRadius;
      UpdatePosition();

      break;
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
      if (event->button.clicks == 2) {
        if (m_DoubleClickCallback) {
          glm::vec2 clickPosition(event->motion.x, event->motion.y);
          glm::vec2 viewportClickPosition = clickPosition - camera.GetViewportMin();
          m_LookAt = camera.GetWorldPosition(m_DoubleClickCallback(viewportClickPosition));
          UpdatePosition();
          PLOG_DEBUG << "New camera center: [" << m_LookAt.x << ", " << m_LookAt.y << ", "
                     << m_LookAt.z << "]";
        } else {
          PLOG_WARNING << "No registered double click callback for Orbit Controller!";
        }
      }
      break;
    }

    default:
      break;
    }
  }
}

void TrackBallController::Update(Camera &camera, int64_t deltaTime, const SDL_Event *event) {
  if (event) {
    const SDL_Keymod modState = SDL_GetModState();

    switch (event->type) {
    case SDL_EVENT_MOUSE_MOTION: {
      const uint32_t buttonState = event->motion.state;
      float dx = static_cast<float>(event->motion.xrel);
      float dy = static_cast<float>(event->motion.yrel);

      glm::vec3 toCamera = m_Position - m_LookAt;
      glm::vec3 right = glm::normalize(glm::cross(toCamera, m_Up));
      glm::vec3 up = glm::normalize(glm::cross(right, toCamera));

      // Pan with right click and drag or shift + left click and drag
      if ((buttonState & SDL_BUTTON_RMASK) ||
          ((modState & SDL_KMOD_SHIFT) && (buttonState & SDL_BUTTON_LMASK))) {
        const float panScale = 0.005f;
        float dist = panScale * glm::length(toCamera);
        dx *= dist;
        dy *= dist;

        glm::vec3 pan = right * dx + up * dy;
        m_LookAt += pan;
        m_Position += pan;
        break;
      }

      // Roll on middle mouse click and drag or ctrl + left mouse click and drag
      if ((buttonState & SDL_BUTTON_MMASK) ||
          ((modState & SDL_KMOD_CTRL) && (buttonState & SDL_BUTTON_LMASK))) {
        auto viewportSize = camera.GetViewportSize();
        float window_width = viewportSize.x;
        float window_height = viewportSize.y;

        float x2 = static_cast<float>(event->motion.x);
        float y2 = static_cast<float>(event->motion.y);

        float x1 = x2 - dx;
        float y1 = y2 - dy;

        // Start point of drag normalized to coords [-1, 1]
        x1 = 2.0f * (x1 / window_width) - 1.0f;
        y1 = -(2.0f * (y1 / window_height) - 1.0f);

        // End point of drag normalized to coords [-1, 1]
        x2 = 2.0f * (x2 / window_width) - 1.0f;
        y2 = -(2.0f * (y2 / window_height) - 1.0f);

        // Compute the change in angle between the start and end point of the drag
        // relative to the center of the screen
        float dTheta = atan2(y2, x2) - atan2(y1, x1);

        m_Up = m_Up * glm::angleAxis(dTheta, glm::normalize(toCamera));
        break;
      }

      // Rotate with left click and drag
      if (buttonState & SDL_BUTTON_LMASK) {
        const float rotateScale = 0.01f;
        dx *= rotateScale;
        dy *= rotateScale;

        glm::quat yaw = glm::angleAxis(dx, m_Up);
        glm::quat pitch = glm::angleAxis(-dy, right);

        glm::vec3 newDir = toCamera * (pitch * yaw);
        m_Position = m_LookAt + newDir;

        m_Up = glm::normalize(glm::cross(-newDir, right));
        break;
      }
      break;
    }

    // Change orbit radius with mouse scroll
    case SDL_EVENT_MOUSE_WHEEL: {
      if (modState & SDL_KMOD_SHIFT) {
        // Shift + mouse wheel is reserved for changing fov/orthographic extent
        break;
      }

      const float minRadius = 0.15f;
      const float radiusScale = 1.1f;
      const float invRadiusScale = 1.0f / radiusScale;
      float delta = 1.0f;
      if (event->wheel.y > 0) {
        delta *= invRadiusScale;
      } else {
        delta *= radiusScale;
      }

      glm::vec3 toCamera = m_Position - m_LookAt;
      float newRadius = glm::length(toCamera) * delta;

      m_Position =
          m_LookAt + glm::normalize(toCamera) * (newRadius > minRadius ? newRadius : minRadius);
      break;
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
      if (event->button.clicks == 2) {
        if (m_DoubleClickCallback) {
          glm::vec3 toCamera = m_Position - m_LookAt;
          glm::vec2 clickPosition(event->motion.x, event->motion.y);
          PLOG_WARNING << "Click Position: " << clickPosition.x << " " << clickPosition.y;
          PLOG_WARNING << "Viewport Min: " << camera.GetViewportMin().x << " "
                       << camera.GetViewportMin().y;
          glm::vec2 viewportClickPosition = clickPosition - camera.GetViewportMin();
          PLOG_WARNING << "Viewport Click Position: " << viewportClickPosition.x << " "
                       << viewportClickPosition.y;
          m_LookAt = camera.GetWorldPosition(m_DoubleClickCallback(clickPosition));
          m_Position = m_LookAt + toCamera;
          PLOG_DEBUG << "New camera center: [" << m_LookAt.x << ", " << m_LookAt.y << ", "
                     << m_LookAt.z << "]";
        } else {
          PLOG_WARNING << "No registered double click callback for Trackball Controller!";
        }
      }
      break;
    }

    default:
      break;
    }
  }
}

glm::vec3 Camera::GetWorldPosition(glm::vec3 screenCoordsDepth) {
  // Convert pixel (x, y) to NDC [-1, 1]
  glm::vec2 viewportSize = GetViewportSize();
  glm::vec2 screenCoords = glm::vec2(screenCoordsDepth);
  glm::vec2 ndc = (screenCoords / viewportSize) * 2.0f - 1.0f;
  glm::vec4 clipSpaceCoord(ndc, screenCoordsDepth.z, 1.0f);

  // Back-project to world space
  glm::mat4 invViewProj = glm::inverse(m_Projection->GetMatrix() * m_Controller->GetMatrix());
  glm::vec4 worldSpaceCoord = invViewProj * clipSpaceCoord;

  // Perspective division
  if (worldSpaceCoord.w != 0.0f) {
    worldSpaceCoord /= worldSpaceCoord.w;
  }

  return glm::vec3(worldSpaceCoord);
}