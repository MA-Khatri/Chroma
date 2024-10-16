#pragma once

#include <glm/glm.hpp>
#include<glm/gtc/type_ptr.hpp>


#define EPSILON 1e-3f

inline bool Close(const glm::vec3& u, const glm::vec3& v)
{
	return glm::all(glm::epsilonEqual(u, v, EPSILON));
}

inline bool Close(const float& u, const float& v)
{
	return abs(u - v) < EPSILON;
}