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
	vec4 cameraPositionAndViewportHeight;
} scene;

layout(set = 1, binding = 0) uniform ObjectUBO {
	mat4 model;
	mat4 normal;
} object;

void main() {
	gl_Position = scene.viewProj * object.model * vec4(a_Position, 1.0);
	gl_PointSize = 5.0;

	fragColor = a_Color;
}