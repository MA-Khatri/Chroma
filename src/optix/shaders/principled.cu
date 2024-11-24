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
	__forceinline__ __device__ float Schlick(float3 a, float3 b)
	{
		float ab = 1.0f - abs(dot(a, b));
		return ab * ab * ab * ab * ab; /* I.e., (1 - |a dot b|)^5 */
	}

	__forceinline__ __device__ float F_D(float3 w, float3 n, float3 h, float3 w_in, float roughness)
	{
		float hwin = dot(h, w_in);
		float F_D90 = 0.5f + 2.0f * roughness * hwin * hwin;
		float s = Schlick(n, w);
		return 1.0f + (F_D90 - 1.0f) * s;
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
		float s = Schlick(n, w);
		return 1.0f + (roughness * hwin2 - 1.0f) * s;
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
		return (1.0f - sbtData.subsurface) * BaseDiffuse(prd, sbtData) 
			+ sbtData.subsurface * Subsurface(prd, sbtData);
	}

	__forceinline__ __device__ float3 MetalFresnel(float3 baseColor, float3 h, float3 w_in)
	{
		return baseColor + (1 - baseColor) * Schlick(h, w_in);
	}

	__forceinline__ __device__ float2 Alphas(float roughness, float anisotropic)
	{
		float a_min = 0.0001f;
		float aspect = sqrt(1.0f - 0.9f * anisotropic);
		float roughness2 = roughness * roughness;

		float a_x = max(a_min, roughness2 / aspect);
		float a_y = max(a_min, roughness2 * aspect);

		return make_float2(a_x, a_y);
	}

	__forceinline__ __device__ float MetalGGX(float3 h, float2 a)
	{
		/* h should (already) be in local shading frame! */
		float d = (h.x * h.x) / (a.x * a.x)
			+ (h.y * h.y) / (a.y * a.y) 
			+ h.z * h.z;
		float denom = M_PIf * a.x * a.y * d * d;
		return 1.0f / denom;
	}

	__forceinline__ __device__ float MetalGeometry(float3 w, float2 a)
	{
		/* Note: w should (already) be in local space! */
		float wax = w.x * a.x;
		float way = w.y * a.y;
		float delta = 0.5f * (-1.0f + sqrt(1.0f + ((wax * wax + way * way) / (w.z * w.z))));
		return 1.0f / (1.0f + delta);
	}

	__forceinline__ __device__ float3 Metal(PRD_Radiance& prd, const SBTData& sbtData)
	{
		float2 a = Alphas(sbtData.roughness, sbtData.anisotropic);

		float3 F_m = MetalFresnel(sbtData.baseColor, prd.h, prd.in_direction);
		float D_m = MetalGGX(prd.basis.Canonical(prd.h), a);
		float G_m = MetalGeometry(prd.basis.Canonical(prd.out_direction), a) 
			* MetalGeometry(prd.basis.Canonical(prd.in_direction), a);

		//float3 F_m = MetalFresnel(sbtData.baseColor, prd.h, prd.in_direction);
		//float D_m = MetalGGX(prd.h, a);
		//float G_m = MetalGeometry(prd.out_direction, a)
		//	* MetalGeometry(prd.in_direction, a);

		return F_m * D_m * G_m / (4.0f * abs(dot(prd.basis.w, prd.out_direction)));
	}


	__forceinline__ __device__ float3 Eval(PRD_Radiance& prd, float3 indir, float3 outdir)
	{
		const SBTData& sbtData = *prd.sbtData;
		const int3 index = sbtData.index[prd.primID];

		PRD_Radiance tempPRD;
		tempPRD.basis = prd.basis;
		tempPRD.h = prd.h;
		tempPRD.in_direction = indir;
		tempPRD.out_direction = outdir;

		float3 diffuseColor = (1.0f - sbtData.specularTransmission) 
			* (1.0f - sbtData.metallic) 
			* Diffuse(tempPRD, sbtData);

		float3 metallicColor = (1.0f - sbtData.specularTransmission * (1.0f - sbtData.metallic)) * Metal(prd, sbtData);

		float3 finalColor = diffuseColor + metallicColor;

		/* If this is the first intersection of the ray, set the albedo and normal */
		if (prd.depth == 0)
		{
			prd.albedo = finalColor;
			prd.normal = prd.basis.w;
		}

		return finalColor;
	}


	__forceinline__ __device__ float3 Sample(PRD_Radiance& prd)
	{
		const SBTData& sbtData = *prd.sbtData;

		float diffuseWeight = (1.0f - sbtData.metallic) * (1.0f - sbtData.specularTransmission);
		float metalWeight = (1.0f - sbtData.specularTransmission * (1.0f - sbtData.metallic));
		float glassWeight = (1.0f - sbtData.metallic) * sbtData.specularTransmission;
		float clearcoatWeight = 0.25f * sbtData.clearcoat;

		return prd.basis.Local(prd.random.RandomOnUnitCosineHemisphere());
		/* TODO: sampling for metals, other... */
	}


	__forceinline__ __device__ float PDF(PRD_Radiance& prd, float3 w)
	{
		return max(dot(w, prd.basis.w), 0.0f) * M_1_PIf;
		/* TODO: other than just assuming cosine hemisphere... */
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