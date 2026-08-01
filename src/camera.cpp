#include "camera.hpp"
#include "imgui.h"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <glm/ext/vector_float2.hpp>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>
#include <limits>

constexpr float SLIDER_WIDTH = 120.0f;
constexpr float DROPDOWN_WIDTH = 160.0f;

const std::unordered_map<Projection::Type, std::string> Projection::m_EnumToStringMap = {
    {Projection::Type::Perspective, "Perspective"},
    {Projection::Type::Orthographic, "Orthographic"},
    {Projection::Type::Unknown, "Unknown"}};

const std::unordered_map<std::string, Projection::Type> Projection::m_StringToEnumMap = {
    {"Perspective", Projection::Type::Perspective},
    {"Orthographic", Projection::Type::Orthographic},
    {"Unknown", Projection::Type::Unknown}};

const std::unordered_map<CameraController::Type, std::string> CameraController::m_EnumToStringMap =
    {{CameraController::Type::FreeFly, "Free Fly"},
     {CameraController::Type::Orbit, "Orbit"},
     {CameraController::Type::TrackBall, "Track Ball"},
     {CameraController::Type::Scanner, "Scanner"},
     {CameraController::Type::Unknown, "Unknown"}};

const std::unordered_map<std::string, CameraController::Type> CameraController::m_StringToEnumMap =
    {{"Free Fly", CameraController::Type::FreeFly},
     {"Orbit", CameraController::Type::Orbit},
     {"Track Ball", CameraController::Type::TrackBall},
     {"Scanner", CameraController::Type::Scanner},
     {"Unknown", CameraController::Type::Unknown}};

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
        if (m_FarClip > m_MaxClip)
          m_FarClip = m_MaxClip;
        if (m_FarClip < m_NearClip + m_MinClipDiff)
          m_FarClip = m_NearClip + m_MinClipDiff;

        PLOG_VERBOSE << "Adjusted far clip to: " << m_FarClip;
      }

      // Alt + mouse wheel for near clip adjustment
      else if (modState & SDL_KMOD_ALT) {
        m_NearClip += event->wheel.y;
        if (m_NearClip < m_MinClip)
          m_NearClip = m_MinClip;
        if (m_NearClip > m_FarClip - m_MinClipDiff)
          m_NearClip = m_FarClip - m_MinClipDiff;

        PLOG_VERBOSE << "Adjusted near clip to: " << m_NearClip;
      }

      // Shift + mouse wheel for FOV (zoom in/out)
      else if (modState & SDL_KMOD_SHIFT) {
        m_VFOV -= event->wheel.y;
        if (m_VFOV < m_MinVFOV)
          m_VFOV = m_MinVFOV;
        if (m_VFOV > m_MaxVFOV)
          m_VFOV = m_MaxVFOV;

        PLOG_VERBOSE << "Adjusted vertical field of view (vfov) to: " << m_VFOV;
      }
      break;
    }

    default:
      break;
    }
  }
}

void PerspectiveProjection::GetGuiElements() {
  ImGui::SeparatorText("Perspective Projection");
  ImGui::PushItemWidth(SLIDER_WIDTH);
  {
    ImGui::DragFloat("Vertical FoV (Shift + Scroll)", &m_VFOV, 0.1f, m_MinVFOV, m_MaxVFOV);

    ImGui::DragFloat("Far Clip (Ctrl + Scroll)", &m_FarClip, m_MinClipDiff, m_MinClip, m_MaxClip);
    if (m_FarClip > m_MaxClip)
      m_FarClip = m_MaxClip;
    if (m_FarClip < m_NearClip + m_MinClipDiff)
      m_FarClip = m_NearClip + m_MinClipDiff;

    ImGui::DragFloat("Near Clip (Alt + Scroll)", &m_NearClip, m_MinClipDiff, m_MinClip, m_MaxClip);
    if (m_NearClip < m_MinClip)
      m_NearClip = m_MinClip;
    if (m_NearClip > m_FarClip - m_MinClipDiff)
      m_NearClip = m_FarClip - m_MinClipDiff;
  }
  ImGui::PopItemWidth();
}

void OrthographicProjection::Update(Camera &camera, int64_t deltaTime, const SDL_Event *event) {
  if (event) {
    switch (event->type) {
    case SDL_EVENT_MOUSE_WHEEL: {
      const SDL_Keymod modState = SDL_GetModState();

      // Ctrl + mouse wheel for far clip adjustment
      if (modState & SDL_KMOD_CTRL) {
        m_FarClip += event->wheel.y;
        if (m_FarClip > m_MaxClip)
          m_FarClip = m_MaxClip;
        if (m_FarClip < m_NearClip + m_MinClipDiff)
          m_FarClip = m_NearClip + m_MinClipDiff;

        PLOG_VERBOSE << "Adjusted far clip to: " << m_FarClip;
      }

      // Alt + mouse wheel for near clip adjustment
      else if (modState & SDL_KMOD_ALT) {
        m_NearClip += event->wheel.y;
        if (m_NearClip < m_MinClip)
          m_NearClip = m_MinClip;
        if (m_NearClip > m_FarClip - m_MinClipDiff)
          m_NearClip = m_FarClip - m_MinClipDiff;

        PLOG_VERBOSE << "Adjusted near clip to: " << m_NearClip;
      }

      // Shift + mouse wheel for orthographic extent
      else if (modState & SDL_KMOD_SHIFT) {
        m_VerticalExtent -= event->wheel.y;
        if (m_VerticalExtent < m_VerticalExtentMin)
          m_VerticalExtent = m_VerticalExtentMin;
        if (m_VerticalExtent > m_VerticalExtentMax)
          m_VerticalExtent = m_VerticalExtentMax;

        PLOG_VERBOSE << "Adjusted orthographic vertical extent to: " << m_VerticalExtent;
      }
      break;
    }

    default:
      break;
    }
  }
}

void OrthographicProjection::GetGuiElements() {
  ImGui::SeparatorText("Orthographic Projection");
  ImGui::PushItemWidth(SLIDER_WIDTH);
  {
    ImGui::DragFloat("Vertical Extent (Shift + Scroll)", &m_VerticalExtent, m_VerticalExtentMin,
                     m_VerticalExtentMin, m_VerticalExtentMax);

    ImGui::DragFloat("Far Clip (Ctrl + Scroll)", &m_FarClip, m_MinClipDiff, m_MinClip, m_MaxClip);
    if (m_FarClip > m_MaxClip)
      m_FarClip = m_MaxClip;
    if (m_FarClip < m_NearClip + m_MinClipDiff)
      m_FarClip = m_NearClip + m_MinClipDiff;

    ImGui::DragFloat("Near Clip (Alt + Scroll)", &m_NearClip, m_MinClipDiff, m_MinClip, m_MaxClip);
    if (m_NearClip < m_MinClip)
      m_NearClip = m_MinClip;
    if (m_NearClip > m_FarClip - m_MinClipDiff)
      m_NearClip = m_FarClip - m_MinClipDiff;
  }
  ImGui::PopItemWidth();
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
        if (m_Pitch > m_PitchMax)
          m_Pitch = m_PitchMax;
        if (m_Pitch < m_PitchMin)
          m_Pitch = m_PitchMin;

        // Clamp yaw from 0-360 degrees
        m_Yaw = std::fmod(m_Yaw, m_YawMax);
        if (m_Yaw < 0.0f)
          m_Yaw = m_YawMax - m_Yaw;

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
        m_MoveUp = pressed;
        break;
      case SDLK_E:
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

void FreeFlyController::GetGuiElements() {
  ImGui::SeparatorText("Free Fly Controller");
  ImGui::PushItemWidth(SLIDER_WIDTH);
  {
    float position[3] = {m_Position.x, m_Position.y, m_Position.z};
    ImGui::DragFloat3("Position (WASD & Q/E to go Up/Down)", position, 1.0f);
    glm::vec3 newPosition(position[0], position[1], position[2]);
    if (m_Position != newPosition) {
      m_Position = newPosition;
      UpdateLookAt();
    }

    float newPitch = m_Pitch;
    ImGui::DragFloat("Pitch (LMB + Drag)", &newPitch, 1.0f, m_PitchMin, m_PitchMax);
    if (newPitch != m_Pitch) {
      m_Pitch = newPitch;
      UpdateLookAt();
    }

    float newYaw = m_Yaw;
    ImGui::DragFloat("Yaw (LMB + Drag)", &newYaw, 1.0f, 0.0f, m_YawMax);
    if (newYaw != m_Yaw) {
      m_Yaw = newYaw;
      UpdateLookAt();
    }
  }
  ImGui::PopItemWidth();
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
      if ((event->motion.state & SDL_BUTTON_LMASK) && (event->button.clicks == 2)) {
        if (m_DoubleClickCallback) {
          // Get click position relative to viewport
          ImVec2 mousePosImVec = ImGui::GetMousePos();
          glm::vec2 mousePosAbs = glm::vec2(mousePosImVec.x, mousePosImVec.y);
          glm::vec2 clickPosition = mousePosAbs - camera.GetViewportMin();

          glm::vec3 candidate = camera.GetWorldPosition(m_DoubleClickCallback(clickPosition));
          if (glm::any(glm::isnan(candidate))) {
            PLOG_DEBUG << "No valid depth at click position!";
            break;
          }
          m_LookAt = candidate;
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

void OrbitController::GetGuiElements() {
  ImGui::SeparatorText("Orbit Controller");
  ImGui::PushItemWidth(SLIDER_WIDTH);
  {
    float newRadius = m_Radius;
    ImGui::DragFloat("Radius (Scroll)", &newRadius, 1.0f, m_MinRadius, m_MaxRadius);
    if (newRadius != m_Radius) {
      m_Radius = newRadius;
      UpdatePosition();
    }

    float center[3] = {m_LookAt.x, m_LookAt.y, m_LookAt.z};
    ImGui::DragFloat3("Center (Double LMB)", center, 1.0f);
    glm::vec3 newCenter(center[0], center[1], center[2]);
    if (m_LookAt != newCenter) {
      m_LookAt = newCenter;
      UpdatePosition();
    }

    ImGui::Text("Rotate: LMB + Drag");
    ImGui::Text("Pan: (RMB or Shift + LMB) + Drag");
  }
  ImGui::PopItemWidth();
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
      m_ToCameraNormalized = glm::normalize(toCamera);
      glm::vec3 right = glm::cross(m_ToCameraNormalized, m_Up);
      glm::vec3 up = glm::cross(right, m_ToCameraNormalized);

      // Pan with right click and drag or shift + left click and drag
      if ((buttonState & SDL_BUTTON_RMASK) ||
          ((modState & SDL_KMOD_SHIFT) && (buttonState & SDL_BUTTON_LMASK))) {
        const float panScale = 0.005f;
        float dist = panScale * m_Radius;
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

        m_Up = m_Up * glm::angleAxis(dTheta, m_ToCameraNormalized);
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
        m_ToCameraNormalized = glm::normalize(newDir);
        m_Position = m_LookAt + newDir;

        m_Up = glm::normalize(glm::cross(-newDir, right));
        break;
      }
      break;
    }

    // Change orbit radius with mouse scroll
    case SDL_EVENT_MOUSE_WHEEL: {
      Uint16 other_mods = modState & ~(SDL_KMOD_NUM | SDL_KMOD_CAPS);
      if (other_mods != SDL_KMOD_NONE) {
        // Scroll + any mod state (except num/caps lock) may be reserved for other controls!
        break;
      }

      float delta = 1.0f;
      if (event->wheel.y > 0) {
        delta *= m_InvRadiusScale;
      } else {
        delta *= m_RadiusScale;
      }

      glm::vec3 toCamera = m_Position - m_LookAt;
      float newRadius = glm::length(toCamera) * delta;
      m_Radius = newRadius > m_MinRadius ? newRadius : m_MinRadius;
      m_Radius = m_Radius > m_MaxRadius ? m_MaxRadius : m_Radius;

      m_ToCameraNormalized = glm::normalize(toCamera);
      m_Position = m_LookAt + m_ToCameraNormalized * m_Radius;
      break;
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
      if ((event->motion.state & SDL_BUTTON_LMASK) && (event->button.clicks == 2)) {
        if (m_DoubleClickCallback) {
          glm::vec3 toCamera = m_Position - m_LookAt;
          m_ToCameraNormalized = glm::normalize(toCamera);

          // Get click position relative to viewport
          ImVec2 mousePosImVec = ImGui::GetMousePos();
          glm::vec2 mousePosAbs = glm::vec2(mousePosImVec.x, mousePosImVec.y);
          glm::vec2 clickPosition = mousePosAbs - camera.GetViewportMin();

          glm::vec3 candidate = camera.GetWorldPosition(m_DoubleClickCallback(clickPosition));
          if (glm::any(glm::isnan(candidate))) {
            PLOG_DEBUG << "No valid depth at click position!";
            break;
          }

          m_LookAt = candidate;
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

void TrackBallController::GetGuiElements() {
  ImGui::SeparatorText("Track Ball Controller");
  ImGui::PushItemWidth(SLIDER_WIDTH);
  {
    float newRadius = m_Radius;
    ImGui::DragFloat("Radius (Scroll)", &newRadius, 1.0f, m_MinRadius, m_MaxRadius);
    if (newRadius != m_Radius) {
      m_Position = m_LookAt + newRadius * m_ToCameraNormalized;
      m_Radius = newRadius;
    }

    float center[3] = {m_LookAt.x, m_LookAt.y, m_LookAt.z};
    ImGui::DragFloat3("Center (Double LMB)", center, 1.0f, 0.0f, 0.0f, "%.1f");
    glm::vec3 newCenter(center[0], center[1], center[2]);
    if (m_LookAt != newCenter) {
      m_LookAt = newCenter;
      m_Position = m_LookAt + m_Radius * m_ToCameraNormalized;
    }

    ImGui::Text("Rotate: LMB + Drag");
    ImGui::Text("Roll: (MMB or Ctrl + LMB) + Drag");
    ImGui::Text("Pan: (RMB or Shift + LMB) + Drag");
  }
  ImGui::PopItemWidth();
}

glm::vec3 Camera::GetWorldPosition(glm::vec3 screenCoordsDepth) {
  // Invalid new depth
  if (glm::any(glm::isnan(screenCoordsDepth)) || screenCoordsDepth.z == 1.0f) {
    return glm::vec3(std::numeric_limits<float>::quiet_NaN());
  }

  // Convert pixel (x, y) to NDC [-1, 1]
  glm::vec2 viewportSize = GetViewportSize();
  glm::vec2 screenCoords = glm::vec2(screenCoordsDepth);
  glm::vec2 ndc = (screenCoords / viewportSize) * 2.0f - 1.0f;
  ndc *= glm::vec2(1.0f, -1.0f); // Flip vertically to account for Vulkan convention
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

void Camera::GetGuiElements() {
  ImGui::PushItemWidth(DROPDOWN_WIDTH);
  {
    // Options to set/select controller and projection types
    Projection::Type cProjectionType = m_Projection->GetType();
    if (ImGui::BeginCombo("Projection Type", m_Projection->GetTypeString().c_str())) {
      for (int n = 0; n < static_cast<int>(Projection::Type::Unknown); n++) {
        Projection::Type type = static_cast<Projection::Type>(n);
        const bool isSelected = (cProjectionType == type);
        if (ImGui::Selectable(Projection::GetTypeString(type).c_str(), isSelected)) {
          cProjectionType = type;
        }
        if (isSelected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    if (m_Projection->GetType() != cProjectionType) {
      PLOG_DEBUG << "Switching to projection type: " << Projection::GetTypeString(cProjectionType);
      if (cProjectionType == Projection::Type::Perspective) {
        m_Projection = std::make_shared<PerspectiveProjection>();
      } else if (cProjectionType == Projection::Type::Orthographic) {
        m_Projection = std::make_shared<OrthographicProjection>();
      } else {
        PLOG_ERROR << "Invalid selected projection type: "
                   << Projection::GetTypeString(cProjectionType);
      }
    }

    CameraController::Type cControllerType = m_Controller->GetType();
    auto callback = m_Controller->GetDoubleClickCallback();
    if (ImGui::BeginCombo("Controller Type", m_Controller->GetTypeString().c_str())) {
      for (int n = 0; n < static_cast<int>(CameraController::Type::Unknown); n++) {
        CameraController::Type type = static_cast<CameraController::Type>(n);
        if (type == CameraController::Type::Scanner) {
          // Cannot choose scanner controller type!
          continue;
        }
        const bool isSelected = (cControllerType == type);
        if (ImGui::Selectable(CameraController::GetTypeString(type).c_str(), isSelected)) {
          cControllerType = type;
        }
        if (isSelected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    if (m_Controller->GetType() != cControllerType) {
      PLOG_DEBUG << "Switching to camera controller type: "
                 << CameraController::GetTypeString(cControllerType);
      if (cControllerType == CameraController::Type::FreeFly) {
        m_Controller = std::make_shared<FreeFlyController>();
        m_Controller->RegisterDoubleClickCallback(callback);
      } else if (cControllerType == CameraController::Type::Orbit) {
        m_Controller = std::make_shared<OrbitController>();
        m_Controller->RegisterDoubleClickCallback(callback);
      } else if (cControllerType == CameraController::Type::TrackBall) {
        m_Controller = std::make_shared<TrackBallController>();
        m_Controller->RegisterDoubleClickCallback(callback);
      } else {
        PLOG_ERROR << "Invalid selected camera controller type: "
                   << CameraController::GetTypeString(cControllerType);
      }
    }
  }
  ImGui::PopItemWidth();

  m_Controller->GetGuiElements();
  m_Projection->GetGuiElements();
}