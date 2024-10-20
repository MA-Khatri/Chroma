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
		if (sbtData.roughness > 0.0f) N = normalize(N + prd.random.RandomOnUnitSphere() * sbtData.roughness);
		
		/* Determine if ray is entering/exiting */
		const float3 w_out = -rayDir;
		float cos_theta_i = dot(w_out, N);

		float eta1, eta2 = 1.0f;
		float3 transmittance = make_float3(1.0f);
		if (cos_theta_i > 0.0f)
		{
			/* Ray is entering */
			eta1 = sbtData.etaIn;
			eta2 = sbtData.etaOut;
		}
		else
		{
			/* Ray is exiting, apply Beer's law */
			transmittance = expf(-sbtData.extinction * optixGetRayTmax());
			eta1 = sbtData.etaOut;
			eta2 = sbtData.etaIn;
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
			prd.radiance *= sbtData.reflectionColor * transmittance;
		}
		else
		{
			/* Refract */
			const float3 w_in = w_t;
			const float3 bhp = BackHitPosition(N);
			prd.origin = bhp;
			prd.direction = w_in;
			prd.radiance *= sbtData.refractionColor * transmittance;
		}

		/* If this is the first intersection of the ray, set the albedo and normal */
		if (prd.depth == 0)
		{
			prd.albedo = sbtData.reflectionColor;
			prd.normal = N;
		}
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}
}