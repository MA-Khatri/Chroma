#version 450

layout(location = 0) in vec3 v_Position;
layout(location = 1) in vec3 v_Normal;
layout(location = 2) in vec3 v_Color;
layout(location = 3) in vec2 v_TexCoord;

layout(set = 0, binding = 0) uniform SceneUBO {
	mat4 view;
	mat4 proj;
	mat4 viewProj;
	vec4 cameraPositionAndViewportHeight;
} scene;

layout(location = 0) out vec4 outColor;

void main() {
	float ambient = 0.2;
	float diffuse = 0.5;
	float specular = 0.1;
	float exponent = 16;

	vec3 cameraPosn = scene.cameraPositionAndViewportHeight.xyz;
	vec3 lightDir = normalize(cameraPosn - v_Position);
	vec3 reflectDir = reflect(-lightDir, v_Normal);

	float diffuseContrib = clamp(dot(lightDir, v_Normal), 0, 1);
	float specularContrib = pow(max(dot(lightDir, reflectDir), 0.0), exponent);

	float lc = ambient + diffuse * diffuseContrib + specular * specularContrib;

	vec3 diffuseColor = v_Color;

	outColor = vec4(diffuseColor * vec3(lc), 1);
}