#include "utils.cuh"

/* 
 * Disney's Principled BSDF implementation, based on https://cseweb.ucsd.edu/~tzli/cse272/wi2023/homework1.pdf
 * and some of the references listed within it. Note, I have a few notational differences to the math
 * presented in that assignment. E.g., my w_in and w_out are opposite of what they use.
 * 
 * The original paper for Disney's principled BRDF can be found here: 
 * https://disneyanimation.com/publications/physically-based-shading-at-disney/
 */

namespace otx
{
	__forceinline__ __device__ float F_D(float3 w, float3 n, float3 h, float3 w_in, float roughness)
	{
		float hwin = dot(h, w_in);
		float F_D90 = 0.5f + 2.0f * roughness * hwin * hwin;
		float s = 1.0f - abs(dot(n, w));
		float s2 = s * s;
		return 1.0f + (F_D90 - 1.0f) * s2 * s2 * s; /* I.e., pow(s, 5.0f) */
	}

	__forceinline__ __device__ float3 BaseDiffuse(PRD_Radiance& prd, const SBTData& sbtData)
	{
		float3 n = prd.basis.w;
		float3 h = prd.h;
		float3 w_in = prd.in_direction;
		float roughness = sbtData.roughness;

		return (sbtData.baseColor * M_1_PIf) 
			* F_D(prd.out_direction, n, h, w_in, roughness) 
			* F_D(prd.in_direction, n, h, w_in, roughness)
			* abs(dot(prd.basis.w, prd.in_direction));
	}

	__forceinline__ __device__ float F_SS(float3 w, float3 n, float3 h, float3 w_in, float roughness)
	{
		float hwin = dot(h, w_in);
		float hwin2 = hwin * hwin;
		float s = 1.0f - abs(dot(n, w));
		float s2 = s * s;
		return 1.0f + (roughness * hwin2 - 1.0f) * s2 * s2 * s;
	}

	/* Approximation of sub-surface scattering using Lommel-Seeliger law */
	__forceinline__ __device__ float3 Subsurface(PRD_Radiance& prd, const SBTData& sbtData)
	{
		float3 n = prd.basis.w;
		float3 h = prd.h;
		float3 w_in = prd.in_direction;
		float3 w_out = prd.out_direction;
		float roughness = sbtData.roughness;

		float nwout = abs(dot(n, w_out));
		float nwin = abs(dot(n, w_in));

		return (1.25f * sbtData.baseColor * M_1_PIf)
			* (F_SS(w_out, n, h, w_in, roughness)
				* F_SS(w_in, n, h, w_in, roughness)
				* ((1.0f / (nwout + nwin)) - 0.5f)
				+ 0.5f)
			* nwin;
	}

	__forceinline__ __device__ float3 Diffuse(PRD_Radiance& prd, const SBTData& sbtData)
	{
		return (1.0f - sbtData.subsurface) * BaseDiffuse(prd, sbtData) + sbtData.subsurface * Subsurface(prd, sbtData);
	}


	__forceinline__ __device__ float3 Sample(PRD_Radiance& prd)
	{
		return prd.basis.Local(prd.random.RandomOnUnitCosineHemisphere());
	}


	__forceinline__ __device__ float3 Eval(PRD_Radiance& prd, float3 indir, float3 outdir)
	{
		const SBTData& sbtData = *prd.sbtData;
		const int3 index = sbtData.index[prd.primID];

		/* Default diffuse color if no diffuse texture */
		float3 diffuseColor = Diffuse(prd, sbtData);

		/* === Sample diffuse texture === */
		float2 tc = TexCoord(prd.uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			diffuseColor *= make_float3(tex.x, tex.y, tex.z);
		}

		/* If this is the first intersection of the ray, set the albedo and normal */
		if (prd.depth == 0)
		{
			prd.albedo = diffuseColor;
			prd.normal = prd.basis.w;
		}

		return diffuseColor * max(dot(indir, prd.basis.w), 0.0f) * M_1_PIf;
	}


	__forceinline__ __device__ float PDF(PRD_Radiance& prd, float3 w)
	{
		return max(dot(w, prd.basis.w), 0.0f) * M_1_PIf;
	}


	extern "C" __global__ void __closesthit__radiance()
	{
		PRD_Radiance& prd = *getPRD<PRD_Radiance>();
		prd.sbtData = (const SBTData*)optixGetSbtDataPointer();
		const SBTData& sbtData = *prd.sbtData;
		prd.Sample = CALLABLE_PRINCIPLED_SAMPLE;
		prd.Eval = CALLABLE_PRINCIPLED_EVAL;
		prd.PDF = CALLABLE_PRINCIPLED_PDF;

		prd.primID = optixGetPrimitiveIndex();
		const int3 index = sbtData.index[prd.primID];
		prd.uv = optixGetTriangleBarycentrics();
		float3 outDir = -optixGetWorldRayDirection();

		/* === Compute normal === */
		/* Use shading normal if available, else use geometry normal */
		const float3& v0 = sbtData.position[index.x];
		const float3& v1 = sbtData.position[index.y];
		const float3& v2 = sbtData.position[index.z];
		float3 N = (sbtData.normal)
			? InterpolateNormals(prd.uv, sbtData.normal[index.x], sbtData.normal[index.y], sbtData.normal[index.z])
			: cross(v1 - v0, v2 - v0);

		/* Compute world-space normal and normalize */
		N = normalize(optixTransformNormalFromObjectToWorldSpace(N));

		/* Face forward normal */
		if (dot(outDir, N) < 0.0f) N = -N;

		/* Update the hit position */
		prd.origin = FrontHitPosition(N);

		/* Update the basis for this intersection */
		prd.basis = OrthonormalBasis(N);

		/* Generate a new sample direction (in_direction) */
		prd.out_direction = prd.in_direction;
		prd.in_direction = Sample(prd);
		prd.h = normalize(prd.in_direction + prd.out_direction);
		prd.specular = false; // MAY NEED TO EDIT THIS!!!

		/* Update throughput */
		float3 bsdf = Eval(prd, prd.in_direction, prd.out_direction);
		float pdf = PDF(prd, prd.in_direction);
		prd.throughput *= bsdf / pdf;
		prd.pdf *= pdf;

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