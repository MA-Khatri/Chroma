#include <optix_device.h>

#include "../launch_params.h"

namespace otx
{
	/* Launch parameters in constant memory, filled in by Optix upon optixLaunch */
	extern "C" __constant__ LaunchParams optixLaunchParams;


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


	/* =============== */
	/* === Helpers === */
	/* =============== */

	/* Compute world position of ray hit */
	extern "C" __device__ float3 HitPosition()
	{
		return optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
	}
	
	/* Compute interpolated normal using barycentric coords */
	extern "C" __device__ float3 InterpolateNormals(const float2& uv, const float3& n0, const float3& n1, const float3& n2)
	{
		return n0 + uv.x * (n1 - n0) + uv.y * (n2 - n0);
	}

	/* Compute interpolated texture coordinate using barycentric coords */
	extern "C" __device__ float2 TexCoord(const float2& uv, const float2& v0, const float2& v1, const float2& v2)
	{
		return (1.0f - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
	}

	/* ===================== */
	/* === Radiance Rays === */
	/* ===================== */
	extern "C" __global__ void __closesthit__radiance()
	{
		const MeshSBTData& sbtData = *(const MeshSBTData*)optixGetSbtDataPointer();
		const int primID = optixGetPrimitiveIndex();
		const int3 index = sbtData.index[primID];
		float2 uv = optixGetTriangleBarycentrics();
		float3 rayDir = optixGetWorldRayDirection();

		/* === Compute normal === */
		/* Use shading normal if available, else use geometry normal */
		const float3& v0 = sbtData.position[index.x];
		const float3& v1 = sbtData.position[index.y];
		const float3& v2 = sbtData.position[index.z];
		const float3& n0 = sbtData.normal[index.x];
		const float3& n1 = sbtData.normal[index.y];
		const float3& n2 = sbtData.normal[index.z];

		float3 Ng = cross(v1 - v0, v2 - v0);
		float3 Ns = (sbtData.normal) ? InterpolateNormals(uv, n0, n1, n2) : Ng;
		
		///* Face forward and normalize normals */
		//if (dot(rayDir, Ng) > 0.0f) Ng = -Ng;
		//Ng = normalize(Ng);

		//if (dot(rayDir, Ns) > 0.0f) Ns = -Ns;
		//Ns = normalize(Ns);

		/* Compute world-space normal and normalize */
		Ns = normalize(optixTransformNormalFromObjectToWorldSpace(Ns));

		/* If no texture, default to white for diffuse color */
		float3 diffuseColor = make_float3(1.0f);

		/* === Sample texture(s) === */
		float2 tc = TexCoord(uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			diffuseColor = make_float3(tex.x, tex.y, tex.z);
		}

		/* === Compute shadow === */
		const float3 surfPosn = HitPosition();
		const float3 lightPosn = make_float3(100.0f, 100.0f, 100.0f); /* Hard coded light position (for now) */
		const float3 lightDir = lightPosn - surfPosn;

		/* Trace shadow ray*/
		float3 lightVisibility = make_float3(0.5f);
		uint32_t u0, u1;
		packPointer(&lightVisibility, u0, u1);
		optixTrace(
			optixLaunchParams.traversable,
			surfPosn + 1e-3f * Ns,
			lightDir,
			1e-3f, /* tmin */
			1.0f-1e-3f, /* tmax -- in terms of lightDir length */
			0.0f, /* ray time */
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
			SHADOW_RAY_TYPE, /* SBT offset */
			RAY_TYPE_COUNT, /* SBT stride */
			SHADOW_RAY_TYPE, /* missSBT index */
			u0, u1 /* packed pointer to our PRD */
		);


		/* Calculate shading */
		const float ambient = 0.2f;
		const float diffuse = 0.5f;
		const float specular = 0.1f;
		const float exponent = 16.0f;

		const float3 reflectDir = reflect(rayDir, Ns);
		const float diffuseContrib = clamp(dot(-rayDir, Ns), 0.0f, 1.0f);
		const float specularContrib = pow(max(dot(-rayDir, reflectDir), 0.0f), exponent);
		const float lc = ambient + diffuse * diffuseContrib + specular * specularContrib;

		/* === Set data === */
		float3& prd = *(float3*)getPRD<float3>();
		//prd = diffuseColor * (lightVisibility * lc);
		prd = diffuseColor * lc;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO
	}

	extern "C" __global__ void __miss__radiance()
	{
		float3& prd = *(float3*)getPRD<float3>();
		prd = optixLaunchParams.clearColor;
	}


	/* =================== */
	/* === Shadow rays === */
	/* =================== */
	extern "C" __global__ void __closesthit__shadow()
	{
		/* Not going to be used... */
	}

	extern "C" __global__ void __anyhit__shadow()
	{
		/* Not going to be used... */
	}

	extern "C" __global__ void __miss__shadow()
	{
		/* Nothing was hit so the light is visible */
		float3& prd = *(float3*)getPRD<float3>();
		prd = make_float3(1.0f);
	}


	/*
	 * The primary ray gen program where camera rays are generated and fired into the scene
	 */
	extern "C" __global__ void __raygen__renderFrame()
	{
		/* Get pixel position */
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		/* Get the camera from launchParams */
		const auto& camera = optixLaunchParams.camera;

		/* The per-ray data is just a color (for now) */
		float3 pixelColorPRD = make_float3(0.0f);

		/* The ints we store the PRD pointer in */
		uint32_t u0, u1;
		packPointer(&pixelColorPRD, u0, u1);

		/* Normalized screen plane position in [0, 1]^2 */
		const float2 screen = make_float2(ix + 0.5f, iy + 0.5f) / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);

		/* Generate ray direction */
		float3 rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);

		/* Launch ray */
		optixTrace(
			optixLaunchParams.traversable,
			camera.position,
			rayDir,
			0.0f, /* tMin */
			1e20f, /* tMax */
			0.0f, /* ray time */
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT, /* OPTIX_RAY_FLAG_NONE */
			RADIANCE_RAY_TYPE, /* SBT offset */
			RAY_TYPE_COUNT, /* SBT stride */
			RADIANCE_RAY_TYPE, /* miss SBT index */
			u0, u1 /* packed pointer to our PRD */
		);

		const int r = int(255.99f*pixelColorPRD.x);
		const int g = int(255.99f * pixelColorPRD.y);
		const int b = int(255.99f * pixelColorPRD.z);

		/* Convert to 32-bit RGBA value, explicitly setting alpha to 0xff */
		const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

		/* Write to the frame buffer */
		const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
		optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
	}

} /* namespace otx */