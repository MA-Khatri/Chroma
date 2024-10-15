#include "utils.cuh"

/*
 * Glass material implementation partially based on: 
 * https://github.com/nvpro-samples/optix_advanced_samples/blob/master/src/optixGlass/glass.cu
 */

namespace otx
{
	extern "C" __global__ void __closesthit__radiance()
	{
		const SBTData& sbtData = *(const SBTData*)optixGetSbtDataPointer();
		PRD_radiance& prd = *getPRD<PRD_radiance>();

		/* === THESE SHOULD LATER BE MATERIAL PARAMS === */
		const float etaIn = 1.45f;
		const float etaOut = 1.0f;
		const float3 reflectionColor = sbtData.reflectionColor;
		const float3 refractionColor = sbtData.refractionColor;
		const float3 extinction = make_float3(0.0f);
		/* ============================================= */

		const int primID = optixGetPrimitiveIndex();
		const int3 index = sbtData.index[primID];
		float2 uv = optixGetTriangleBarycentrics();
		float3 rayDir = optixGetWorldRayDirection();

		/* === Compute normal === */
		/* Use shading normal if available, else use geometry normal */
		const float3& v0 = sbtData.position[index.x];
		const float3& v1 = sbtData.position[index.y];
		const float3& v2 = sbtData.position[index.z];
		float3 N = (sbtData.normal)
			? InterpolateNormals(uv, sbtData.normal[index.x], sbtData.normal[index.y], sbtData.normal[index.z]) 
			: cross(v1 - v0, v2 - v0);

		/* Compute world-space normal and normalize */
		N = normalize(optixTransformNormalFromObjectToWorldSpace(N));

		
		/* Determine if ray is entering/exiting */
		const float3 w_out = -rayDir;
		float cos_theta_i = dot(w_out, N);

		float eta1, eta2;
		float3 transmittance = make_float3(1.0f);
		if (cos_theta_i > 0.0f)
		{
			/* Ray is entering */
			eta1 = etaIn;
			eta2 = etaOut;
		}
		else
		{
			/* Ray is exiting, apply Beer's law */
			transmittance = expf(-extinction * optixGetRayTmax());
			eta1 = etaOut;
			eta2 = etaIn;
			cos_theta_i = -cos_theta_i;
			N = -N;
		}

		/* Determine refracted ray and if it was totally internally reflected */
		float3 w_t;
		const bool tir = !refract(w_t, -w_out, N, eta1, eta2);
		const float cos_theta_t = -dot(N, w_t);
		const float R = tir ? 1.0f : fresnel(cos_theta_i, cos_theta_t, eta1, eta2);

		/* Importance sample the Fresnel term */
		const float z = prd.random();
		if (z <= R)
		{
			/* Reflect */
			const float3 w_in = reflect(-w_out, N);
			const float3 fhp = FrontHitPosition(N);
			prd.origin = fhp;
			prd.direction = w_in;
			prd.radiance *= reflectionColor * transmittance;
		}
		else
		{
			/* Refract */
			const float3 w_in = w_t;
			const float3 bhp = BackHitPosition(N);
			prd.origin = bhp;
			prd.direction = w_in;
			prd.radiance *= refractionColor * transmittance;
		}
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}
}