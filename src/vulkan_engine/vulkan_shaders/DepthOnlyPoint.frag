#version 450

layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_Color;

void main() {
  float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
  if (dist >= 0.5) {
    discard;
  }
}
