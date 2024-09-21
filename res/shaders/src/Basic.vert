#version 460

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec2 a_TexCoord;

layout(location = 0) out vec3 fragColor;

void main() 
{
	gl_Position = vec4(a_Position, 1.0);
	fragColor = normalize(a_Normal);
}