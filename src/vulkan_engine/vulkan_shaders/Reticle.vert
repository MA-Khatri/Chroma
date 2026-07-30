#version 450

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec3 a_Color;
layout(location = 3) in vec2 a_TexCoord;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform SceneUBO {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  vec3 cameraPosition;
  vec2 viewportSize;
}
scene;

layout(set = 1, binding = 0) uniform ObjectUBO {
  mat4 model;
  mat4 normal;
}
object;

const float kReticleAspect = 0.75;
const float kReticleVFoV = 9.5; // vertical FOV in degrees

void main() {
  // Recover tan(cameraVFov/2) from the projection matrix.
  // For a standard perspective matrix, proj[1][1] == 1 / tan(vFov/2).
  // abs() guards against the Vulkan Y-flip convention (proj[1][1] negated).
  float projYScale = abs(scene.proj[1][1]);

  // NDC half-height that corresponds to an angular size of kReticleVFoV degrees
  // within the camera's current vertical FOV.
  float ndcHalfHeight = tan(radians(kReticleVFoV) * 0.5) * projYScale;

  // Correct the width so the reticle keeps kReticleAspect (width/height) in
  // screen space, independent of the viewport's own aspect ratio.
  float screenAspect = scene.viewportSize.x / scene.viewportSize.y;
  float ndcHalfWidth = ndcHalfHeight * kReticleAspect / screenAspect;

  vec2 ndcOffset = vec2(a_Position.x * ndcHalfWidth, a_Position.y * ndcHalfHeight);

  gl_Position = vec4(ndcOffset, 0.0, 1.0);
  fragColor = a_Color;
}