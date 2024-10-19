#pragma once

#include <glm/glm.hpp>
#include <vector_math.h>

#include "../common_enums.h"

namespace otx
{
	/* Mesh and material data */
	struct SBTData
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

		float etaIn;
		float etaOut;
		float roughness;
		float3 reflectionColor;
		float3 refractionColor;
		float3 extinction;
	};
	

	struct LaunchParams
	{
		struct {
			float4* colorBuffer;
			float4* normalBuffer;
			float4* albedoBuffer;
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

			/* Specific to thin lens */
			float3 defocusDiskU;
			float3 defocusDiskV;
		} camera;

		float3 cutoffColor; /* radiance color for rays that reach depth limit */
		int maxDepth; /* Max number of ray bounces */

		OptixTraversableHandle traversable;

		int backgroundMode;
		float3 clearColor;
		float3 gradientBottom;
		float3 gradientTop;
		cudaTextureObject_t backgroundTexture;

		bool gammaCorrect; /* Should gamma correction be applied to the final image? */
	};

} /* namspace otx */
