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

		CALLABLE_SAMPLE_BACKGROUND,
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
		float3 emissionColor;
		float3 extinction;
	};
	

	/* MIS Light definition */
	struct MISLight
	{
		/* Index into LightType enum (in common_enums.h) */
		int type;

		/* Emission color */
		float3 emissionColor;

		/* Optional emission texture (mainly just area lights) */
		bool hasTexture;
		cudaTextureObject_t emissionTexture;

		/* Total power of light (i.e., length(emissionColor) * area) */
		float power;

		/* Position of vertex 0 for area lights; origin of light for delta lights */
		float3 p0;

		/* Position of vertex 1 for area lights; direction of light for delta lights */
		float3 p1;

		/* Position of vertex 2 for area lights; x = inner angle, y = outer angle, z = blend mode for delta lights */
		float3 p2;

		/* Light normals for area lights */
		float3 n0;
		float3 n1;
		float3 n2;

		/* Texture coordinates for area lights */
		float2 t0;
		float2 t1;
		float2 t2;

		/* Only used for area lights */
		float area;
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

		int maxDepth; /* Max number of ray bounces. If set to 0, use russian roulette path termination. */

		/* unsigned long long == OptixTraversableHandle -- just not included here since this file is included in non-optix sections */
		unsigned long long traversable; /* Optix traversable handle for top-level scene AS */


		int backgroundMode;
		float3 clearColor;
		float3 gradientBottom;
		float3 gradientTop;
		cudaTextureObject_t backgroundTexture;
		float backgroundRotation;

		bool gammaCorrect; /* Should gamma correction be applied to the final image? */

		/* An array of lights to importance sample */
		MISLight* lights;

		/* The number of importance sampled lights in the scene */
		int nLights; 
	};

} /* namspace otx */
