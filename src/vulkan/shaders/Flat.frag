#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D diffuseSampler;
layout(binding = 2) uniform sampler2D specularSampler;
layout(binding = 3) uniform sampler2D normalSampler;

void main() 
{
	outColor = texture(diffuseSampler, fragTexCoord);
}