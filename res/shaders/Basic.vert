#version 450

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

void main() 
{
	mat4 matrix = ubo.proj * ubo.view * ubo.model;

	gl_Position = matrix * vec4(a_Position, 1.0);

	fragColor = a_Normal;
}