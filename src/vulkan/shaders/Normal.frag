#version 450

// Use with "Solid.vert"

//layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
//layout(location = 2) in vec3 v_CameraPosn;

layout(location = 0) out vec4 outColor;

void main() 
{
	outColor = vec4(v_Normal, 1);
}