#pragma once

#include <string>

#include "math_helpers.h"
#include "common_enums.h"
#include "object.h"
#include "material.h"

/* ======================= */
/* === Light Interface === */
/* ======================= */
class Light
{
	// TODO

protected:
	/* Index into the LightType enum */
	int m_LightType;

	/* The total radiant flux (i.e., power) of the light */
	float m_Power;
};


/* ====================================== */
/* === Background (Environment) Light === */
/* ====================================== */
class BackgroundLight : public Light
{
	BackgroundLight(std::string path);

private:
	/* Path to the background image texture */
	std::string m_TexturePath;

	/* Background texture loaded as a float to enable hdr skyboxes */
	Texture<float> m_BackgroundTexture;
};


/* ========================= */
/* === Area (Mesh) Light === */
/* ========================= */
class AreaLight : public Light
{
	/* 
	 * An area light is effectively just a mesh light. 
	 * Each triangle in a mesh is treated as its own area light.
	 * This makes it easy to sample a point on the triangle, 
	 * calculate the area of the light, calculate its total power, etc.
	 */

public:
	AreaLight(Vertex v0, Vertex v1, Vertex v2, glm::vec3 radiantExitance, glm::mat4 transform = glm::mat4(1.0f), glm::mat4 nTransform = glm::mat4(1.0f));

	struct LightVertex
	{
		float3 posn;
		float3 normal;
		float2 texCoord;
	};

private:
	/* The surface area of the light */
	float m_Area = 0.0f;

	/* Store info about each light vertex */
	LightVertex m_V0;
	LightVertex m_V1;
	LightVertex m_V2;

};

/* Creates a vector of area lights from the triangle mesh of a given scene object and its corresponding material. */
std::vector<std::shared_ptr<Light>> ObjectLight(Object object);


/* ================================ */
/* === Delta (point/spot) Light === */
/* ================================ */
class DeltaLight : public Light
{
	/*
	 * Note: Point lights are effectively just a special case of the spot light,
	 * so these types of lights are combined into a single class.
	 */


	// TODO

private:
	/* Position of the light */
	glm::vec3 m_Position;
	
	/* The direction the light cone is pointed */
	glm::vec3 m_Direction;
	
	/* The angle of the light cone w.r.t. the direction. Angle at which intensity starts to fall off. */
	float m_InnerAngle = M_PIf; 

	/* The angle of the light cone w.r.t. the direction Angle at which intensity reaches 0. */
	float m_OuterAngle = M_PIf;

	/* Index into blend mode, signifies how to blend intensity between inner and outer angles. */
	int m_BlendMode = BLEND_MODE_LINEAR;
};