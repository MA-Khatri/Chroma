#pragma once

#include <glm/glm.hpp>
#include <vector_math.h>

#include "../common_enums.h"

namespace otx
{
	enum CallableIDs
	{
		CALLABLE_LAMBERTIAN_EVAL = 0,
		CALLABLE_LAMBERTIAN_PDF,
		CALLABLE_CONDUCTOR_EVAL,
		CALLABLE_CONDUCTOR_PDF,
		CALLABLE_DIELECTRIC_EVAL,
		CALLABLE_DIELECTRIC_PDF,
		// TODO... add the rest
		CALLABLE_COUNT
	};

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
		int sampler; /* Index into SamplerType enum */
		int nStrata; /* Number of strata along each sampling dimension */
		int integrator; /* Index into IntegratorType enum */

		struct {
			float4* colorBuffer;
			float4* normalBuffer;
			float4* albedoBuffer;
			int2 size; /* Width, height of frame */
			int samples; /* Pixel samples per launch (i.e., call to render) */
			int frameID{ 0 }; /* Current frame count */
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
		int lightSampleCount; /* Number of direct light samples per surface intersection */

		OptixTraversableHandle traversable; /* Optix traversable handle for top-level scene AS */


		int backgroundMode;
		float3 clearColor;
		float3 gradientBottom;
		float3 gradientTop;
		cudaTextureObject_t backgroundTexture;
		float backgroundRotation;

		bool gammaCorrect; /* Should gamma correction be applied to the final image? */
	};

} /* namspace otx */
