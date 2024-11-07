#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector_math.h>

#define EPSILON 1e-4f

inline bool Close(const glm::vec3& u, const glm::vec3& v)
{
	return glm::all(glm::epsilonEqual(u, v, EPSILON));
}

inline bool Close(const float& u, const float& v)
{
	return abs(u - v) < EPSILON;
}

inline float3 ToFloat3(const glm::vec3& v)
{
	return { v.x, v.y, v.z };
}

inline float2 ToFloat2(const glm::vec2& v)
{
	return { v.x, v.y };
}