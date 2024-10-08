#pragma once

#include <glm/glm.hpp>
#include <vector_math.h>


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
		float3* position;
		float3* normal;
		float2* texCoord; 
		int3* index;

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
			int2 size;
		} frame;

		struct {
			float3 position;
			float3 direction;
			float3 horizontal;
			float3 vertical;
			float verticalFOVdeg;
		} camera;

		float3 clearColor;

		OptixTraversableHandle traversable;
	};

} /* namspace otx */
