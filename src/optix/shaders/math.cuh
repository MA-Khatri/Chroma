#pragma once

namespace otx
{
	/* ========================= */
	/* === Orthonormal basis === */
	/* ========================= */
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
	inline __host__ __device__ float2 SampleInUnitDiskConcentric(float2 u)
	{
		/* Map u to [-1, 1]^2 and handle degeneracy at origin */
		float2 uOffset = 2 * u - make_float2(1.0f);
		if (uOffset.x == 0.0f && uOffset.y == 0.0f)
		{
			return make_float2(0.0f);
		}

		/* Apply concentric mapping to point */
		float theta, r;
		if (abs(uOffset.x) > abs(uOffset.y))
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
		float2 d = SampleInUnitDiskConcentric(u);
		float z = sqrt(1 - d.x * d.x - d.y * d.y);
		return make_float3(d.x, d.y, z);
	}
}