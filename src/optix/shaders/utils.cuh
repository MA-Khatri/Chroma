#pragma once

#include <optix_device.h>
#include <cuda_runtime.h>
#include "../launch_params.h"

namespace otx
{
	/* ====================== */
	/* === Shader helpers === */
	/* ====================== */

#define RAY_EPS 1e-5f

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
	static __inline__ __device__ float3 HitPosition()
	{
		return optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
	}

	/* Compute the world hit position offset by a small epsilon along the normal */
	static __inline__ __device__ float3 FrontHitPosition(float3 n)
	{
		return HitPosition() + n * RAY_EPS;
	}

	/* Compute the world hit position offset by a small epsilon along the negative normal */
	static __inline__ __device__ float3 BackHitPosition(float3 n)
	{
		return HitPosition() - n * RAY_EPS;
	}

	/* Compute interpolated normal using barycentric coords */
	static __inline__ __device__ float3 InterpolateNormals(const float2 & uv, const float3 & n0, const float3 & n1, const float3 & n2)
	{
		return n0 + uv.x * (n1 - n0) + uv.y * (n2 - n0);
	}

	/* Compute interpolated texture coordinate using barycentric coords */
	static __inline__ __device__ float2 TexCoord(const float2 & uv, const float2 & v0, const float2 & v1, const float2 & v2)
	{
		return (1.0f - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
	}


	/* Compute effective Fresnel reflectance: https://en.wikipedia.org/wiki/Fresnel_equations */
	static __inline__ __device__ float fresnel(float cos_theta_i, float cos_theta_t, float eta1, float eta2)
	{
		const float rs = (eta1 * cos_theta_i - eta2 * cos_theta_t) / (eta1 * cos_theta_i + eta2 * cos_theta_t);
		const float rt = (eta1 * cos_theta_t - eta2 * cos_theta_i) / (eta1 * cos_theta_t + eta2 * cos_theta_i);
		return 0.5f * (rs * rs + rt * rt);
	}

	/* 
	 * Set refracted ray and return boolean indicating if the result is total internal reflection.
	 * -- w_t = transmitted (or internally reflected) ray that will be set.
	 * -- w_i = incident ray
	 * -- n = surface normal
	 * -- eta1 = exterior index of refraction (wrt the normal)
	 * -- et12 = interior index of refraction (wrt the normal)
	 */
	static __inline__ __device__ bool refract(float3& w_t, float3 w_i, float3 n, float eta1, float eta2)
	{
		float cos_theta_i = dot(-w_i, n);
		float eta = eta2 / eta1;
		float cos_theta_t_2 = 1.0f - eta * eta * (1.0f - cos_theta_i * cos_theta_i);
		w_t = eta * w_i + ((eta * cos_theta_i - sqrt(abs(cos_theta_t_2))) * n);
		return cos_theta_t_2 > 0.0f;
	}

	/* Compute the log of each component of a float3 */
	static __inline__ __device__ float3 logf3(float3 v)
	{
		return make_float3(logf(v.x), logf(v.y), logf(v.z));
	}

	/* ========================== *
	 * === Sampling functions === *
	 * ========================== *
	 * based on: https://pbr-book.org/4ed/Sampling_Algorithms/Sampling_Multidimensional_Functions
	 */

	/* Sample a point on the circumference of a unit circle */
	inline __host__ __device__ float2 SampleOnUnitCircle(float u)
	{
		float angle = u * 2.0f * M_PIf;
		return make_float2(cos(angle), sin(angle));
	}

	/* Sample a point within a unit disk using polar coordinates */
	inline __host__ __device__ float2 SampleInUnitDiskPolar(float2 u)
	{
		float r = sqrtf(u.x);
		float t = 2 * M_PIf * u.y;
		return make_float2(r * cos(t), r * sin(t));
	}

	/* Sample a point within a unit disk using a concentric mapping */
	/* WARNING: There appears to be a bug with this... */
	inline __host__ __device__ float2 SampleInUnitDiskConcentric(float2 u)
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
	inline __host__ __device__ float3 SampleOnUnitSphere(float2 u)
	{
		float z = 1.0f - 2.0f * u.x;
		float r = sqrt(1.0f - z * z);
		float phi = 2.0f * M_PIf * u.y;
		return make_float3(r * cos(phi), r * sin(phi), z);
	}

	/* Sample a point on the surface of a unit hemisphere */
	inline __host__ __device__ float3 SampleOnUnitHemisphere(float2 u)
	{
		float z = u.x;
		float r = sqrt(1.0f - z * z);
		float phi = 2.0f * M_PIf * u.y;
		return make_float3(r * cos(phi), r * sin(phi), z);
	}

	/* Sample a point on the surface of a unit hemisphere with cosine weighting */
	inline __host__ __device__ float3 SampleOnUnitCosineHemisphere(float2 u)
	{
		float2 d = SampleInUnitDiskPolar(u);
		float z = sqrt(1 - d.x * d.x - d.y * d.y);
		return make_float3(d.x, d.y, z);
	}

	/* ==========================================
	 * === Probability Distribution Functions ===
	 * ==========================================
	 * PDFs for different types of samplers
	 */

	inline __host__ __device__ float UnitSpherePDF()
	{
		return 0.07957747154f; /* 0.07957747154 = 1 / (4 * PI) */
	}

	inline __host__ __device__ float UnitHemispherePDF()
	{
		return 0.15915494309f; /* 0.15915494309 = 1 / (2 * PI) */
	}

	inline __host__ __device__ float CosineHemispherePDF(float3 v)
	{
		float cosTheta = sqrt(v.x * v.x + v.y * v.y);
		return cosTheta * M_1_PIf;
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

		/* Transform input vector to local space */
		__host__ __device__ float3 Local(const float3& a) const
		{
			return a.x * u + a.y * v + a.z * w;
		}

	public:
		float3 u, v, w;
	};


	/* ============================================= *
	 * ===  Permuted Congruential Generator(PCG) === *
	 * ============================================= *
	 * Random number generation and random sampling using PCG: www.pcg-random.org
	 */
	struct PCG
	{
		uint32_t state;

		inline __host__ __device__ PCG()
		{
			/*
			 * intentionally empty so we can use it in device vars that
			 * don't allow dynamic initialization (ie, PRD)
			 */
		}

		inline __host__ __device__ PCG(uint32_t seed)
		{
			Init(seed);
		}

		inline __host__ __device__ void Init(uint32_t seed)
		{
			state = seed;
			NextRandom();
		}

		inline __host__ __device__ void NextRandom()
		{
			state = state * 747796405 + 2891336453;
			state = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
			state = (state >> 22) ^ state;
		}

		/* Alias for RandomValue() */
		inline __host__ __device__ float operator() ()
		{
			return RandomValue();
		}

		/* Return a random value in [0, 1) */
		inline __host__ __device__ float RandomValue()
		{
			NextRandom();
			return state / 4294967295.0; /* 4294967295.0 = 2^32 - 1*/
		}

		/* Return a random point *on the circumference* of a unit circle */
		inline __host__ __device__ float2 RandomOnUnitCircle()
		{
			return SampleOnUnitCircle(RandomValue());
		}

		/* Return a random point *in* a unit circle (disk) */
		inline __host__ __device__ float2 RandomInUnitDisk()
		{
			return SampleInUnitDiskPolar(make_float2(RandomValue(), RandomValue()));
		}

		/* Return a random point on the surface of a unit sphere */
		inline __host__ __device__ float3 RandomOnUnitSphere()
		{
			return SampleOnUnitSphere(make_float2(RandomValue(), RandomValue()));
		}

		/* Return a random point on the surface of a unit hemisphere */
		inline __host__ __device__ float3 RandomOnUnitHemisphere()
		{
			return SampleOnUnitHemisphere(make_float2(RandomValue(), RandomValue()));
		}

		/* Return a cosine weighted random point on the surface of a unit hemisphere */
		inline __host__ __device__ float3 RandomOnUnitCosineHemisphere()
		{
			return SampleOnUnitCosineHemisphere(make_float2(RandomValue(), RandomValue()));
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
		Random random; /* Random number generator and its state */
		int depth; /* Recursion depth */

		/* Shading state */
		bool done; /* boolean allowing for early termination, e.g. if ray gets fully absorbed/misses */
		float3 radiance; /* I.e., pixel color */
		float3 normal; /* Surface normal of first intersection */
		float3 albedo; /* Diffuse color of first intersection */
		float3 origin; /* Store the *next* ray's origin */
		float3 direction; /* Store the *next* ray's direction */
	};

	struct PRD_Shadow
	{
		float3 radiance; /* Light's radiance */
		bool reachedLight; /* Did the shadow ray reach the light? */
	};
}