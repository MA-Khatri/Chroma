#version 450

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;   // unused
layout(location = 2) in vec3 a_Color;
layout(location = 3) in vec2 a_TexCoord; // unused

layout(location = 0) out vec3 v_Color;
layout(location = 1) out vec3 v_Position;
layout(location = 2) out vec3 v_ClearColor;

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

	v_Color = a_Color;
	v_Position = (object.model * vec4(a_Position, 1.0)).xyz;
	v_ClearColor = object.normal[0].xyz; // The clear color is stored in the first column of the normal matrix!
}