#version 450

layout(location = 0) in vec3 v_Color;
layout(location = 1) in vec3 v_Position;
layout(location = 2) in vec3 v_CameraPosn;

layout(location = 0) out vec4 outColor;

void main() 
{
	// How fast is falloff?
	float coeff = 0.01;

	// Where does color falloff start?
	float start = 20.0;

	// At what distance are lines discarded?
	float end = 50.0;


	float dist = length(v_Position - v_CameraPosn);
	float scale = exp(-coeff * (dist - start));

	if (dist < start) 
		outColor = vec4(v_Color, 1.0);
	else if (dist < end)
		outColor = vec4(v_Color, scale * 1.0);
	else 
		discard;
}