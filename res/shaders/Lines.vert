#version 450

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Color;

layout(location = 0) out vec3 v_Color;
layout(location = 1) out vec3 v_Position;
layout(location = 2) out vec3 v_CameraPosn; // Note: camera posn, dir, and clear color are technically not varying, but we're just passing them along
layout(location = 3) out vec3 v_ClearColor;


layout(push_constant) uniform CameraMatrices {
	mat4 view;
	mat4 proj;
} camera;


layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 normal;
} object;


void main() 
{
	gl_Position = camera.proj * camera.view * object.model * vec4(a_Position, 1.0);

	v_Color = a_Color;
	v_Position = (object.model * vec4(a_Position, 1.0)).xyz;
	v_CameraPosn = inverse(camera.view)[3].xyz;
	v_ClearColor = object.normal[0].xyz; // The clear color is stored in the first column of the normal matrix...
}