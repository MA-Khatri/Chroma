#pragma once

#include <glm/glm.hpp>

namespace otx
{
	struct MeshSBTData
	{
		glm::vec3* vertex;
		glm::ivec3* index;
		glm::vec3* normal;
		glm::vec2* texCoord; 
	};
	

	struct LaunchParams
	{
		int frameID{ 0 };

		struct {
			uint32_t* colorBuffer;
			glm::ivec2 size;
		} frame;

		struct {
			glm::vec3 position;
			glm::vec3 direction;
			glm::vec3 horizontal;
			glm::vec3 vertical;
			float verticalFOVdeg;
		} camera;

		OptixTraversableHandle traversable;
	};

} /* namspace otx */
