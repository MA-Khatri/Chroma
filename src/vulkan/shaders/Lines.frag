#version 450

layout(location = 0) in vec3 v_Color;
layout(location = 1) in vec3 v_Position;
layout(location = 2) in vec3 v_CameraPosn;
layout(location = 3) in vec3 v_ClearColor;

layout(location = 0) out vec4 outColor;

void main() 
{
	// How fast is falloff?
	float coeff = 0.05;

	// Where does color falloff start?
	float start = 10.0;

	float dist = length(v_Position - v_CameraPosn);
	float scale = 1.0 - exp(-coeff * (dist - start));

	if (dist < start) 
		outColor = vec4(v_Color, 1.0);
	else if (scale < 0.99)
		outColor = vec4(mix(v_Color, v_ClearColor, scale), 1.0);
	else 
		discard;
}