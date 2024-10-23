#include "utils.cuh"

namespace otx
{
	extern "C" __global__ void __closesthit__radiance()
	{
		const SBTData& sbtData = *(const SBTData*)optixGetSbtDataPointer();
		PRD_Radiance& prd = *getPRD<PRD_Radiance>();

		const int primID = optixGetPrimitiveIndex();
		const int3 index = sbtData.index[primID];
		float2 uv = optixGetTriangleBarycentrics();
		float3 rayDir = optixGetWorldRayDirection();

		/* === Compute normal === */
		/* Use shading normal if available, else use geometry normal */
		const float3& v0 = sbtData.position[index.x];
		const float3& v1 = sbtData.position[index.y];
		const float3& v2 = sbtData.position[index.z];
		float3 N = (sbtData.normal) ? InterpolateNormals(uv, sbtData.normal[index.x], sbtData.normal[index.y], sbtData.normal[index.z]) : cross(v1 - v0, v2 - v0);

		/* Compute world-space normal and normalize */
		N = normalize(optixTransformNormalFromObjectToWorldSpace(N));

		/* Face forward normal */
		if (dot(rayDir, N) > 0.0f) N = -N;

		/* Default diffuse color if no diffuse texture */
		float3 diffuseColor = sbtData.reflectionColor;

		/* === Sample diffuse texture === */
		float2 tc = TexCoord(uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			diffuseColor = make_float3(tex.x, tex.y, tex.z);
		}

		/* === Set ray data for next trace call === */
		/* Determine reflected ray origin and direction */
		OrthonormalBasis basis = OrthonormalBasis(N);
		prd.direction = basis.Local(prd.random.RandomOnUnitCosineHemisphere());
		prd.origin = FrontHitPosition(N);

		/* Update the primary ray path color */
		prd.radiance *= diffuseColor;

		/* If this is the first intersection of the ray, set the albedo and normal */
		if (prd.depth == 0)
		{
			prd.albedo = diffuseColor;
			prd.normal = N;
		}




		/* === Direct Light Sampling === */
		
		float3 shadowRayOrg = prd.origin;

		for (int i = 0; i < optixLaunchParams.lightSampleCount; i++)
		{
			/* Pick a light to sample... */
			// TODO

			/* For now we just pick a point on the surface of the 3x3 cornell box light */
			float r1 = prd.random();
			float r2 = prd.random();
			float3 lightSamplePosition = make_float3(r1 * 6.0f - 3.0f, r2 * 6.0f - 3.0f, 9.99f);
			float3 lightSampleDirection = lightSamplePosition - shadowRayOrg;
			float3 lightNormalDirection = make_float3(0.0f, 0.0f, -1.0f);
			float3 normalizedLightSampleDirection = normalize(lightSampleDirection);

			/* Initialize a shadow ray... */
			PRD_Shadow shadowRay;
			shadowRay.radiance = make_float3(0.0f);
			shadowRay.reachedLight = false;

			/* Launch the shadow ray towards the selected light */
			uint32_t s0, s1;
			packPointer(&shadowRay, s0, s1);
			optixTrace(
				optixLaunchParams.traversable,
				shadowRayOrg, /* I.e., last hit position of the primary ray path */
				normalize(lightSampleDirection),
				0.0f, /* shadowRayOrg should already be offset */
				length(lightSampleDirection) - RAY_EPS,
				0.0f, /* ray time */
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
				RAY_TYPE_SHADOW,
				RAY_TYPE_COUNT,
				RAY_TYPE_SHADOW,
				s0, s1
			);

			if (shadowRay.reachedLight)
			{
				prd.totalRadiance += prd.radiance * shadowRay.radiance * CosineHemispherePDF(basis.Canonical(normalizedLightSampleDirection));
				prd.nLightPaths++;
			}
		}
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}
}