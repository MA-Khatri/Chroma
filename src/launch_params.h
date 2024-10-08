#pragma once

#include <glm/glm.hpp>

namespace otx
{
	/* Ray types */
	enum
	{
		RADIANCE_RAY_TYPE = 0,
		SHADOW_RAY_TYPE,
		RAY_TYPE_COUNT
	};

	struct MeshSBTData
	{
		glm::vec3* position;
		glm::vec3* normal;
		glm::vec2* texCoord; 
		glm::ivec3* index;

		bool hasDiffuseTexture;
		cudaTextureObject_t diffuseTexture;
		bool hasSpecularTexture;
		cudaTextureObject_t specularTexture;
		bool hasNormalTexture;
		cudaTextureObject_t normalTexture;
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

		glm::vec3 clearColor;

		OptixTraversableHandle traversable;
	};

} /* namspace otx */
