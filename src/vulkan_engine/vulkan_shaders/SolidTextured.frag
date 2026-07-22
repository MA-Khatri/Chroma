#version 450

layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_Color;
layout(location = 3) in vec2 v_TexCoord;
layout(location = 4) in vec3 v_CameraPosn;

layout(set = 1, binding = 1) uniform sampler2D diffuseSampler;
layout(set = 1, binding = 2) uniform sampler2D specularSampler;
layout(set = 1, binding = 3) uniform sampler2D normalSampler;

layout(location = 0) out vec4 outColor;

void main() {
	float ambient = 0.2;
	float diffuse = 0.5;
	float specular = 0.1;
	float exponent = 16;

	vec3 lightDir = normalize(v_CameraPosn - v_Position);
	vec3 reflectDir = reflect(-lightDir, v_Normal);

	float diffuseContrib = clamp(dot(lightDir, v_Normal), 0, 1);
	float specularContrib = pow(max(dot(lightDir, reflectDir), 0.0), exponent);

	float lc = ambient + diffuse * diffuseContrib + specular * specularContrib;

	// vec3 diffuseColor = v_Color;
	vec3 diffuseColor = texture(diffuseSampler, v_TexCoord).rgb;
	
	vec3 specularColor = vec3(1.0);
	// vec3 specularColor = texture(specularSampler, v_TexCoord).rgb;

	outColor = vec4(diffuseColor * specularColor * vec3(lc), 1);
}