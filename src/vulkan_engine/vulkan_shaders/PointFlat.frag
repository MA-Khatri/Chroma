#version 450

layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_Color;

layout(set = 0, binding = 0) uniform SceneUBO {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  vec3 cameraPosition;
  vec2 viewportSize;
}
scene;

layout(location = 0) out vec4 outColor;

void main() {
  float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
  if (dist < 0.5) {
    outColor = vec4(v_Color, 1);
  } else {
    discard;
  }
}