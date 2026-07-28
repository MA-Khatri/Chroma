#version 450

layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_Color;
layout(location = 3) in vec2 v_TexCoord;

layout(set = 0, binding = 0) uniform SceneUBO {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  vec4 cameraPositionAndViewportHeight;
}
scene;

layout(location = 0) out vec4 outColor;

void main() { outColor = vec4(v_Color, 1); }