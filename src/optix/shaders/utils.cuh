#pragma once

#include <optix_device.h>
#include <cuda_runtime.h>
#include "../launch_params.h"

namespace otx
{
	/* ====================== */
	/* === Shader helpers === */
	/* ====================== */

#define RAY_EPS 1e-4f

	/*
	 * To communicate between programs, we pass a pointer to per-ray data (PRD)
	 * which we represent with two ints. The helpers below allow us to encode/decode
	 * the pointer as two ints.
	 */
	static __forceinline__ __device__
		void* unpackPointer(uint32_t i0, uint32_t i1)
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*> (uptr);
		return ptr;
	}

	static __forceinline__ __device__
		void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
	{
		const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template<typename T>
	static __forceinline__ __device__ T* getPRD()
	{
		const uint32_t u0 = optixGetPayload_0();
		const uint32_t u1 = optixGetPayload_1();
		return reinterpret_cast<T*>(unpackPointer(u0, u1));
	}


	/* Compute world position of (current) ray hit */
	static __forceinline__ __device__ float3 HitPosition()
	{
		return optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
	}

	/* Compute the world hit position offset by a small epsilon along the normal */
	static __forceinline__ __device__ float3 FrontHitPosition(float3 n)
	{
		return HitPosition() + n * RAY_EPS;
	}

	/* Compute the world hit position offset by a small epsilon along the negative normal */
	static __forceinline__ __device__ float3 BackHitPosition(float3 n)
	{
		return HitPosition() - n * RAY_EPS;
	}

	/* Compute interpolated normal using barycentric coords */
	static __forceinline__ __device__ float3 InterpolateNormals(const float2 & uv, const float3 & n0, const float3 & n1, const float3 & n2)
	{
		return n0 + uv.x * (n1 - n0) + uv.y * (n2 - n0);
	}

	/* Compute interpolated texture coordinate using barycentric coords */
	static __forceinline__ __device__ float2 TexCoord(const float2 & uv, const float2 & v0, const float2 & v1, const float2 & v2)
	{
		return (1.0f - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
	}


	/* Compute effective Fresnel reflectance: https://en.wikipedia.org/wiki/Fresnel_equations */
	static __forceinline__ __device__ float fresnel(float cos_theta_i, float cos_theta_t, float eta1, float eta2)
	{
		const float rs = (eta1 * cos_theta_i - eta2 * cos_theta_t) / (eta1 * cos_theta_i + eta2 * cos_theta_t);
		const float rt = (eta1 * cos_theta_t - eta2 * cos_theta_i) / (eta1 * cos_theta_t + eta2 * cos_theta_i);
		return 0.5f * (rs * rs + rt * rt);
	}

	/* 
	 * Set refracted ray and return boolean indicating if the ray actually refracted (else it internally reflected).
	 * -- w_t = transmitted (or internally reflected) ray that will be set.
	 * -- w_i = incident ray
	 * -- n = surface normal
	 * -- eta1 = exterior index of refraction (wrt the normal)
	 * -- et12 = interior index of refraction (wrt the normal)
	 */
	static __forceinline__ __device__ bool refract(float3& w_t, float3 w_i, float3 n, float eta1, float eta2)
	{
		float cos_theta_i = dot(-w_i, n);
		float eta = eta2 / eta1;
		float cos_theta_t_2 = 1.0f - eta * eta * (1.0f - cos_theta_i * cos_theta_i);
		w_t = eta * w_i + ((eta * cos_theta_i - sqrt(abs(cos_theta_t_2))) * n);
		return cos_theta_t_2 > 0.0f;
	}

	/* Compute the log of each component of a float3 */
	static __forceinline__ __device__ float3 logf3(float3 v)
	{
		return make_float3(logf(v.x), logf(v.y), logf(v.z));
	}

	/* Determine if 2 floats are within some small epsilon of each other */
	static __forceinline__ __device__ bool close(float a, float b)
	{
		return abs(a - b) < RAY_EPS;
	}


	/* Determine if all components of 2 float3's are within some small epsilon of each other */
	static __forceinline__ __device__ bool close(float3 a, float3 b)
	{
		return (close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z));
	}


	/* Determine the weight using the power heuristic with beta = 2 */
	static __forceinline__ __device__ float powerHeuristic(float a, float b)
	{
		return a * a / (a * a + b * b);
	}

	/* ========================== *
	 * === Sampling functions === *
	 * ========================== *
	 * based on: https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions
	 */

	/* Sample a point on the circumference of a unit circle */
	__forceinline__ __host__ __device__ float2 SampleOnUnitCircle(float u)
	{
		float angle = u * 2.0f * M_PIf;
		return make_float2(cos(angle), sin(angle));
	}

	/* Sample a point within a unit disk using polar coordinates */
	__forceinline__ __host__ __device__ float2 SampleInUnitDiskPolar(float2 u)
	{
		float r = sqrtf(u.x);
		float t = 2 * M_PIf * u.y;
		return make_float2(r * cos(t), r * sin(t));
	}

	/* Sample a point within a unit disk using a concentric mapping */
	/* WARNING: There appears to be a bug with this... */
	__forceinline__ __host__ __device__ float2 SampleInUnitDiskConcentric(float2 u)
	{
		/* Map u to [-1, 1]^2 and handle degeneracy at origin */
		float2 uOffset = 2.0f * u - make_float2(1.0f);
		if (uOffset.x == 0.0f && uOffset.y == 0.0f)
		{
			return make_float2(0.0f);
		}

		/* Apply concentric mapping to point */
		float theta, r;
		if (fabsf(uOffset.x) > fabsf(uOffset.y))
		{
			r = uOffset.x;
			theta = M_PI_4f * (uOffset.y / uOffset.y);
		}
		else
		{
			r = uOffset.y;
			theta = M_PI_2f - M_PI_4f * (uOffset.x / uOffset.y);
		}
		return r * make_float2(cos(theta), sin(theta));
	}

	/* Sample a point on the surface of a unit sphere */
	__forceinline__ __host__ __device__ float3 SampleOnUnitSphere(float2 u)
	{
		float z = 1.0f - 2.0f * u.x;
		float r = sqrt(1.0f - z * z);
		float phi = M_2PIf * u.y;
		return make_float3(r * cos(phi), r * sin(phi), z);
	}

	/* Sample a point on the surface of a unit hemisphere */
	__forceinline__ __host__ __device__ float3 SampleOnUnitHemisphere(float2 u)
	{
		float z = u.x;
		float r = sqrt(1.0f - z * z);
		float phi = M_2PIf * u.y;
		return make_float3(r * cos(phi), r * sin(phi), z);
	}

	/* 
	 * Sample a point on the surface of a unit hemisphere with cosine weighting.
	 * Generated by first sampling the unit disk then projecting onto the hemisphere.
	 */
	__forceinline__ __host__ __device__ float3 SampleOnUnitCosineHemisphere(float2 u)
	{
		float2 d = SampleInUnitDiskPolar(u);
		float z = sqrt(1 - d.x * d.x - d.y * d.y);
		return make_float3(d.x, d.y, z);
	}

	/* ========================= *
	 * === Orthonormal basis === *
	 * ========================= *
	 * Create an orthonormal basis using a normal vector to
	 * transform generated samples w.r.t. the basis
	 */
	class OrthonormalBasis
	{
	public:
		__host__ __device__ OrthonormalBasis() {};

		__host__ __device__ OrthonormalBasis(const float3& normal)
		{
			BuildFromW(normal);
		}

		/* Build orthonormal basis from w vector (usually a normal) */
		__host__ __device__ void BuildFromW(const float3& w_in)
		{
			/* Normalize the input vector, just in case */
			w = normalize(w_in);

			/* Create an arbitrary other vector... */
			float3 a = fabsf(w.x) > 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);

			/* Use that vector to generate an orthonormal basis */
			v = normalize(cross(w, a));
			u = cross(w, v);
		}

		/* Transform input vector to local (model) space */
		__host__ __device__ float3 Local(const float3& a) const
		{
			return a.x * u + a.y * v + a.z * w;
		}

		/* Compute the inverse transformation matrix */
		/* based on: https://stackoverflow.com/questions/983999/simple-3x3-matrix-inverse-code-c */
		__host__ __device__ void ComputeInverse()
		{
			float det = u.x * ((v.y * w.z) - (w.y * v.z)) -
				u.y * ((v.x * w.z) - (v.z * w.z)) +
				u.z * ((v.x * w.y) - (v.y * w.x));

			float invdet = 1.0f / det;

			ui.x = ((v.y * w.z) - (w.y * v.z)) * invdet;
			ui.y = ((u.z * w.y) - (u.y * w.z)) * invdet;
			ui.z = ((u.y * v.z) - (u.z * v.y)) * invdet;
			vi.x = ((v.z * w.x) - (v.x * w.z)) * invdet;
			vi.y = ((u.x * w.z) - (u.z * w.x)) * invdet;
			vi.z = ((v.x * u.z) - (u.x * v.z)) * invdet;
			wi.x = ((v.x * w.y) - (w.x * v.y)) * invdet;
			wi.y = ((w.x * u.y) - (u.x * w.y)) * invdet;
			wi.z = ((u.x * v.y) - (v.x * u.y)) * invdet;

			inverseComputed = true;
			/* I never want to do this again... */
		}

		/* Transform input vector back to canonical space */
		__host__ __device__ float3 Canonical(const float3& a)
		{
			if (!inverseComputed) ComputeInverse();

			return normalize(a.x * ui + a.y * vi + a.z * wi);
		}


	public:
		float3 u, v, w; /* can be interpreted as rows of a 3x3 change of basis matrix */
		float3 ui, vi, wi; /* inverse of basis vectors (inverse 3x3 matrix) */
		bool inverseComputed = false;
	};


	/* ============================================= *
	 * ===  Permuted Congruential Generator(PCG) === *
	 * ============================================= *
	 * Random number generation and random sampling using PCG: www.pcg-random.org
	 */
	struct PCG
	{
		/* PCG state used to generate independent random samples */
		uint32_t state;

		/* Index into SampleType enum -- used to determine method for generating 1D/2D samples */
		int samplerType;

		/* Used in some samplers to determine the number of strata to divide the sample domain into, along each dimension */
		int nStrata;

		/* The width of each stratum */
		float stratumWidth;


		__forceinline__ __host__ __device__ PCG()
		{
			/*
			 * intentionally empty so we can use it in device vars that
			 * don't allow dynamic initialization (ie, PRD)
			 */
		}

		__forceinline__ __host__ __device__ PCG(uint32_t seed, int type, int strata)
		{
			Init(seed, type, strata);
		}

		__forceinline__ __host__ __device__ void Init(uint32_t seed, int type, int strata)
		{
			state = seed;
			samplerType = type;
			nStrata = strata;
			stratumWidth = 1.0f / (float)nStrata;
			NextRandom();
		}

		__forceinline__ __host__ __device__ void NextRandom()
		{
			state = state * 747796405 + 2891336453;
			state = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
			state = (state >> 22) ^ state;
		}

		/* Alias for RandomValue() */
		__forceinline__ __host__ __device__ float operator() ()
		{
			return RandomValue();
		}

		/* Return a(n independent) random value in [0, 1) */
		__forceinline__ __host__ __device__ float RandomValue()
		{
			NextRandom();
			return state / 4294967295.0f; /* 4294967295.0 = 2^32 - 1*/
		}

		/* Generate a 1D random sample in [0, 1) using chosen sampler type */
		__forceinline__ __host__ __device__ float RandomSample1D()
		{
			float sample = 0.0f;

			switch (samplerType)
			{
			case SAMPLER_TYPE_INDEPENDENT:
			{
				sample = RandomValue();
				break;
			}
			case SAMPLER_TYPE_STRATIFIED:
			{
				int stratum = (int)(RandomValue() * (float)nStrata); /* Pick a random stratum */
				float offset = (float)stratum * stratumWidth;
				sample = offset + RandomValue() * stratumWidth;
				break;
			}
			case SAMPLER_TYPE_MULTIJITTER:
			{
				// TODO
				// https://graphics.pixar.com/library/MultiJitteredSampling/
				break;
			}
			}

			return sample;
		}

		/* Generate a 2D random sample in [0, 1)^2 using chosen sampler type */
		__forceinline__ __host__ __device__ float2 RandomSample2D()
		{
			float2 sample = make_float2(0.0f);

			switch (samplerType)
			{
			case SAMPLER_TYPE_INDEPENDENT:
			{
				sample = make_float2(RandomValue(), RandomValue());
				break;
			}
			case SAMPLER_TYPE_STRATIFIED:
			{
				int stratum = (int)(RandomValue() * (float)(nStrata * nStrata)); /* Pick a random stratum in 2D */
				float offsetX = (float)(stratum % nStrata) * stratumWidth;
				float offsetY = (float)(stratum / nStrata) * stratumWidth;
				sample = make_float2(offsetX + RandomValue() * stratumWidth, offsetY + RandomValue() * stratumWidth);
				break;
			}
			case SAMPLER_TYPE_MULTIJITTER:
			{
				// TODO
				// https://graphics.pixar.com/library/MultiJitteredSampling/
				break;
			}
			}

			return sample;
		}

		/* Return a random point *on the circumference* of a unit circle */
		__forceinline__ __host__ __device__ float2 RandomOnUnitCircle()
		{
			return SampleOnUnitCircle(RandomSample1D());
		}

		/* Return a random point *in* a unit circle (disk) */
		__forceinline__ __host__ __device__ float2 RandomInUnitDisk()
		{
			return SampleInUnitDiskPolar(RandomSample2D());
		}

		/* Return a random point on the surface of a unit sphere */
		__forceinline__ __host__ __device__ float3 RandomOnUnitSphere()
		{
			return SampleOnUnitSphere(RandomSample2D());
		}

		/* Return a random point on the surface of a unit hemisphere */
		__forceinline__ __host__ __device__ float3 RandomOnUnitHemisphere()
		{
			return SampleOnUnitHemisphere(RandomSample2D());
		}

		/* Return a cosine weighted random point on the surface of a unit hemisphere */
		__forceinline__ __host__ __device__ float3 RandomOnUnitCosineHemisphere()
		{
			return SampleOnUnitCosineHemisphere(RandomSample2D());
		}
	};


	/* ==================== *
	 * ===  Shader Info === *
	 * ==================== *
	 * All information that is passed to our shader.
	 * This has to be at the end of this file since it uses definitions from above.
	 */
	
	/* Launch parameters in constant memory, filled in by Optix upon optixLaunch */
	extern "C" __constant__ LaunchParams optixLaunchParams;

	typedef PCG Random;

	/* === Per-ray data structs === */
	struct PRD_Radiance
	{
		/* Random number generator and its state */
		Random random; 

		/* Recursion depth (of primary ray path) */
		int depth;


		/* === Shading state === */
		
		/* boolean allowing for early termination, e.g. if ray gets fully absorbed/misses */
		bool done;

		/* Primary ray path's cumulative weight (i.e., product of (bsdf * cosine / pdf) from all bounces) */
		float3 throughput;

		/* Product of all pdfs along the ray path */
		float pdf;

		/* Stores resulting color for this sample */
		float3 color; 


		/* Surface normal of first intersection -- used in the denoiser */
		float3 normal;

		/* Diffuse color of first intersection -- used in the denoiser */
		float3 albedo; 
		

		/* Store the position of the intersection. For backwards path tracing (i.e., from camera to lights), the next ray path to trace will be (origin + in_direction * t) */
		float3 origin;

		/* I.e., direction of incoming light to the surface. For backwards path tracing, this is the direction generated by the bsdf's sample() method. */
		float3 in_direction;

		/* I.e., the direction of outgoing light from the surface. */
		float3 out_direction;

		/* ONB of the most recent intersection -- can be used to get normal vector (basis.w) or do model/world transformations */
		OrthonormalBasis basis;

		/* Store a pointer to the sbt data for the most recent intersection */
		const SBTData* sbtData;

		/* Stores the UV coordinates of the intersection */
		float2 uv;
		
		/* Stores the ID of the last hit primitive */
		int primID;

		/* Used for dielectrics to store whether the sampled ray was refracted */
		bool refracted;

		/* The path integrator has a special condition for specular bounces, so we track that here */
		bool specular;

		/* === Store callable function idxs for the most-recent intersection's bsdf interface methods === */

		/* Generates a new sample direction from the bsdf based on the current prd.out_direction */
		int Sample;

		/* 
		 * Evaluates the material's ability to reflect light from indir to outdir (i.e., the BSDF term). Includes cosine term.
		 */
		int Eval;

		/* Computes the relative probability of sampling direction w (given the prd's current prd.out_direction) */
		int PDF;
	};

	struct PRD_Shadow
	{
		float3 throughput;
		float pdf;
		bool reached_light; /* Did the shadow ray reach the light? */
	};
}