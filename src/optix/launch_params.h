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
		
		float3* color; /* Default color used in place of diffuse texture if texture is not present */

		bool hasDiffuseTexture;
		cudaTextureObject_t diffuseTexture;
		bool hasSpecularTexture;
		cudaTextureObject_t specularTexture;
		bool hasNormalTexture;
		cudaTextureObject_t normalTexture;
	};
	

	struct LaunchParams
	{
		struct {
			uint32_t* colorBuffer; /* Where final result is stored */
			float* accumBuffer; /* Where accumulated color is stored before conversion to colorBuffer */
			int2 size; /* Width, height of frame */
			int samples; /* Pixel samples per launch (i.e., call to render) */
			int accumID{ 0 }; /* Current accumulated frame count */
		} frame;

		struct {
			float3 position;
			float3 direction;
			float3 horizontal;
			float3 vertical;
			int projectionMode;
		} camera;

		float3 cutoff_color; /* radiance color for rays that reach depth limit */
		int maxDepth; /* Max number of ray bounces */

		OptixTraversableHandle traversable;
	};

} /* namspace otx */
