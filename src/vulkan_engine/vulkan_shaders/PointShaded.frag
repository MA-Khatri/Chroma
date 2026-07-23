#version 450

layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_Color;

layout(set = 0, binding = 0) uniform SceneUBO {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  vec4 cameraPositionAndViewportHeight;
}
scene;

layout(location = 0) out vec4 outColor;

void main() {
  float dist = distance(gl_PointCoord, vec2(0.5, 0.5));
  if (dist < 0.5) {
    vec3 cameraPosn = scene.cameraPositionAndViewportHeight.xyz;
    vec3 cameraDir = normalize(cameraPosn - v_Position);

    float t = mix(0.5, 1.0, dot(v_Normal, cameraDir));
    outColor = vec4(v_Color * t, 1);
  } else {
    discard;
  }
}