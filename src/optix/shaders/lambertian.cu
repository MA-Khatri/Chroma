#include "utils.cuh"

namespace otx
{
	__forceinline__ __device__ float3 Sample(PRD_Radiance& prd)
	{
		return prd.basis.Local(prd.random.RandomOnUnitCosineHemisphere());
	}

	__forceinline__ __device__ float3 Eval(PRD_Radiance& prd, float3 indir, float3 outdir)
	{
		const SBTData& sbtData = *prd.sbtData;
		const int3 index = sbtData.index[prd.primID];

		/* Default diffuse color if no diffuse texture */
		float3 diffuseColor = sbtData.reflectionColor;

		/* === Sample diffuse texture === */
		float2 tc = TexCoord(prd.uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			diffuseColor = make_float3(tex.x, tex.y, tex.z);
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
		/* I.e., cosine hemisphere pdf */
		return max(dot(w, prd.basis.w), 0.0f) * M_1_PIf;
	}


	extern "C" __global__ void __closesthit__radiance()
	{
		PRD_Radiance& prd = *getPRD<PRD_Radiance>();
		prd.sbtData = (const SBTData*)optixGetSbtDataPointer();
		const SBTData& sbtData = *prd.sbtData;
		prd.Sample = CALLABLE_LAMBERTIAN_SAMPLE;
		prd.Eval = CALLABLE_LAMBERTIAN_EVAL;
		prd.PDF = CALLABLE_LAMBERTIAN_PDF;
		
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
		prd.specular = false;

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


	extern "C" __device__ float3 __direct_callable__eval(PRD_Radiance& prd, float3 indir, float3 outdir)
	{
		return Eval(prd, indir, outdir);
	}


	extern "C" __device__ float __direct_callable__pdf(PRD_Radiance& prd, float3 w)
	{
		return PDF(prd, w);
	}
}