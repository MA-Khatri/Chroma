#include "utils.cuh"

/*
 * Glass material implementation partially based on: 
 * https://github.com/nvpro-samples/optix_advanced_samples/blob/master/src/optixGlass/glass.cu
 */

namespace otx
{
	__forceinline__ __device__ float3 Sample(PRD_Radiance& prd)
	{
		const SBTData& sbtData = *prd.sbtData;
		float3 N = prd.basis.w;

		/* Determine if ray is entering/exiting */
		float cos_theta_i = dot(prd.out_direction, N);

		float eta1, eta2 = 1.0f;
		float3 transmittance = make_float3(1.0f);
		if (cos_theta_i > 0.0f)
		{ /* Ray is entering */
			eta1 = sbtData.etaIn;
			eta2 = sbtData.etaOut;
		}
		else
		{ /* Ray is exiting */
			eta1 = sbtData.etaOut;
			eta2 = sbtData.etaIn;
			cos_theta_i = -cos_theta_i;
			N = -N;
		}

		/* Determine refracted ray and if it was totally internally reflected */
		float3 w_t;
		const bool tir = !refract(w_t, -prd.out_direction, N, eta1, eta2);
		const float cos_theta_t = -dot(N, w_t);
		const float R = tir ? 1.0f : fresnel(cos_theta_i, cos_theta_t, eta1, eta2);

		/* Importance sample the Fresnel term using Russian Roulette */
		const float z = prd.random();
		if (z <= R)
		{ /* Reflect */
			return reflect(-prd.out_direction, N);
		}
		else
		{ /* Refract */
			return w_t;
		}
	}

	__forceinline__ __device__ float3 Eval(PRD_Radiance& prd, float3 indir, float3 outdir)
	{
		/* Note: technically, we should be returning with an inf term but ignore it since the infs in this and the pdf should cancel out */

		const SBTData& sbtData = *prd.sbtData;
		float3 c;

		if (prd.refracted)
		{
			float3 N = prd.basis.w;
			c = sbtData.refractionColor;

			/* Determine if ray is entering/exiting */
			float cos_theta_i = dot(indir, N);

			float eta1, eta2 = 1.0f;
			if (cos_theta_i > 0.0f)
			{ /* Ray is entering */
				eta1 = sbtData.etaIn;
				eta2 = sbtData.etaOut;
			}
			else
			{ /* Ray is exiting */
				eta1 = sbtData.etaOut;
				eta2 = sbtData.etaIn;
				cos_theta_i = -cos_theta_i;
				N = -N;
			}

			/* Determine refracted ray and if it was totally internally reflected */
			float3 w_t;
			const bool tir = !refract(w_t, -indir, N, eta1, eta2);
			const float cos_theta_t = -dot(N, w_t);
			const float R = tir ? 1.0f : fresnel(cos_theta_i, cos_theta_t, eta1, eta2);

			/* The full version... */
			//return abs(cos_theta_i) * ((eta2 * eta2) / (eta1 * eta1)) * (1.0f - R) / abs(cos_theta_i);

			/* The Cosine terms fall out so we can set it to just this: */
			return c * ((eta2 * eta2) / (eta1 * eta1)) * (1.0f - R);
		}
		else
		{
			c = sbtData.reflectionColor;

			/* The full version... */
			//return max(dot(indir, prd.basis.w), 0.0f) * close(reflect(outdir, prd.basis.w), indir) ? 1.0f / max(dot(indir, prd.basis.w), 1e-4f) : 0.0f;

			/* Cosine terms cancel out so we can set it to just: */
			return c * (close(reflect(outdir, prd.basis.w), indir) ? 1.0f : 0.0f);
		}
	}


	__forceinline__ __device__ float PDF(PRD_Radiance& prd, float3 w)
	{
		/* Note: technically, we should be returning inf but we return 1 since the infs in this and the eval should cancel out */
		return close(prd.in_direction, w) ? 1.0f : 0.0f;
	}


	extern "C" __global__ void __closesthit__radiance()
	{
		PRD_Radiance& prd = *getPRD<PRD_Radiance>();
		prd.sbtData = (const SBTData*)optixGetSbtDataPointer();
		const SBTData& sbtData = *prd.sbtData;
		prd.Sample = CALLABLE_DIELECTRIC_SAMPLE;
		prd.Eval = CALLABLE_DIELECTRIC_EVAL;
		prd.PDF = CALLABLE_DIELECTRIC_PDF;

		const int primID = optixGetPrimitiveIndex();
		const int3 index = sbtData.index[primID];
		float2 uv = optixGetTriangleBarycentrics();
		float3 outDir = -optixGetWorldRayDirection();

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

		/* Update the basis for this intersection -- will always be stored pointing outwards! */
		prd.basis = OrthonormalBasis(N);
	
		/* Determine if ray is entering/exiting */
		float cos_theta_i = dot(outDir, N);

		float eta1, eta2 = 1.0f;
		float3 transmittance = make_float3(1.0f);
		if (cos_theta_i > 0.0f)
		{ /* Ray is entering */
			eta1 = sbtData.etaIn;
			eta2 = sbtData.etaOut;
		}
		else
		{ /* Ray is exiting, apply Beer's law */
			transmittance = expf(-sbtData.extinction * optixGetRayTmax());
			eta1 = sbtData.etaOut;
			eta2 = sbtData.etaIn;
			cos_theta_i = -cos_theta_i;
			N = -N;
		}

		/* Determine refracted ray and if it was totally internally reflected */
		float3 w_t;
		const bool tir = !refract(w_t, -outDir, N, eta1, eta2);
		const float cos_theta_t = -dot(N, w_t);
		const float R = tir ? 1.0f : fresnel(cos_theta_i, cos_theta_t, eta1, eta2);

		/* Importance sample the Fresnel term using Russian Roulette */
		const float z = prd.random();
		if (z <= R)
		{ /* Reflect */
			const float3 w_in = reflect(-outDir, N);
			const float3 fhp = FrontHitPosition(N);
			prd.origin = fhp;
			prd.out_direction = prd.in_direction;
			prd.in_direction = w_in;
			prd.refracted = false;
			prd.specular = true;
		}
		else
		{ /* Refract */
			const float3 w_in = w_t;
			const float3 bhp = BackHitPosition(N);
			prd.origin = bhp;
			prd.out_direction = prd.in_direction;
			prd.in_direction = w_in;
			prd.refracted = true;
			prd.specular = true;
		}

		/* Update the throughput */
		float pdf = PDF(prd, prd.in_direction);
		prd.pdf *= pdf;
		prd.throughput *= transmittance * Eval(prd, prd.in_direction, prd.out_direction) / pdf;

		/* If this is the first intersection of the ray, set the albedo and normal */
		if (prd.depth == 0)
		{
			prd.albedo = sbtData.reflectionColor;
			prd.normal = N;
		}

		/* Store the world space positions of the hit triangle vertices */
		prd.p0 = optixTransformPointFromObjectToWorldSpace(v0);
		prd.p1 = optixTransformPointFromObjectToWorldSpace(v1);
		prd.p2 = optixTransformPointFromObjectToWorldSpace(v2);
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}


	extern "C" __device__ float3 __direct_callable__sample(PRD_Radiance & prd)
	{
		return Sample(prd);
	}


	extern "C" __device__ float3 __direct_callable__eval(PRD_Radiance & prd, float3 indir, float3 outdir)
	{
		return Eval(prd, indir, outdir);
	}


	extern "C" __device__ float __direct_callable__pdf(PRD_Radiance & prd, float3 w)
	{
		return PDF(prd, w);
	}
}