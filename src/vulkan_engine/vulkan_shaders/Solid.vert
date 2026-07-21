#version 450

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec3 a_Color;
layout(location = 3) in vec2 a_TexCoord;

layout(location = 0) out vec3 v_Position;
layout(location = 1) out vec3 v_Normal;
layout(location = 2) out vec3 v_Color;
layout(location = 3) out vec2 v_TexCoord;
layout(location = 4) out vec3 v_CameraPosn;

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

	v_Position = (object.model * vec4(a_Position, 1.0)).xyz;
	v_Normal = (object.normal * vec4(a_Normal, 0.0)).xyz;
	v_Color = a_Color;
	v_TexCoord = a_TexCoord;
	v_CameraPosn = scene.cameraPositionAndViewportHeight.xyz;
}