#version 450

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec3 a_Color;
layout(location = 3) in vec2 a_TexCoord; // TODO: consider creating a separate graphics pipeline for
                                         // points that does not take in a texcoord attribute

layout(location = 0) out vec3 v_Position;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec3 v_Color;

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

void main() {
  vec4 worldPos = object.model * vec4(a_Position, 1.0);
  vec4 viewPos = scene.view * worldPos;
  gl_Position = scene.proj * viewPos;

  // proj[1][1] is the standard vertical focal length factor (1.0 / tan(fov_y / 2))
  float fovScalingFactor = scene.proj[1][1];

  const float targetPercent = 0.05; // percent of viewport height

  // Scale up by the projection scale and viewport height, and down by the view-space depth
  gl_PointSize = (targetPercent * scene.viewportSize.y * fovScalingFactor) / abs(viewPos.z);

  // Assign outputs
  v_Position = worldPos.xyz;
  v_Normal = (object.normal * vec4(a_Normal, 0.0)).xyz;
  v_Color = a_Color;
}