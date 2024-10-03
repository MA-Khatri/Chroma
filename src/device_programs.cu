#include <optix_device.h>

#include "launch_params.h" /* Also includes glm */

namespace otx
{
	/* Launch parameters in constant memory, filled in by Optix upon optixLaunch */
	extern "C" __constant__ LaunchParams optixLaunchParams;

	
	/* Ray types */
	enum 
	{
		SURFACE_RAY_TYPE = 0,
		RAY_TYPE_COUNT
	};


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


	extern "C" __device__ glm::vec3 InterpolateNormals(const float2& uv, const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2)
	{
		return n0 + uv.x * (n1 - n0) + uv.y * (n2 - n0);
	}

	/*
	 * Closest hit and any hit programs for radiance-type rays.
	 * Eventually, we will need a pair of these for each ray type 
	 * and geometry type that we want to render.
	 */
	extern "C" __global__ void __closesthit__radiance()
	{
		const MeshSBTData& sbtData = *(const MeshSBTData*)optixGetSbtDataPointer();

		/* Compute normal */
		const int primID = optixGetPrimitiveIndex();
		const glm::ivec3 index = sbtData.index[primID];
		//const glm::vec3& A = sbtData.vertex[index.x];
		//const glm::vec3& B = sbtData.vertex[index.y];
		//const glm::vec3& C = sbtData.vertex[index.z];
		//const glm::vec3 Ng = glm::normalize(glm::cross(B - A, C - A));

		//auto rd = optixGetWorldRayDirection();
		//const float cosDN = 0.2f + 0.8f * fabsf(glm::dot(glm::vec3(rd.x, rd.y, rd.z), Ng));
		//glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
		//prd = cosDN * sbtData.color;

		const glm::vec3& n0 = sbtData.normal[index.x];
		const glm::vec3& n1 = sbtData.normal[index.y];
		const glm::vec3& n2 = sbtData.normal[index.z];
		float2 uv = optixGetTriangleBarycentrics();
		glm::vec3 interpolatedNormal = InterpolateNormals(uv, n0, n1, n2);

		glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
		prd = 1.0f * interpolatedNormal;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO
	}


	/* 
	 * Miss program that gets called for any ray that did not have a valid intersection.
	 */
	extern "C" __global__ void __miss__radiance()
	{
		glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
		prd = glm::vec3(1.0f); /* WHITE */
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
		glm::vec3 pixelColorPRD = glm::vec3(0.0f);

		/* The ints we store the PRD pointer in */
		uint32_t u0, u1;
		packPointer(&pixelColorPRD, u0, u1);

		/* Normalized screen plane position in [0, 1]^2 */
		const glm::vec2 screen = glm::vec2(glm::vec2(ix + 0.5f, iy + 0.5f) / glm::vec2(optixLaunchParams.frame.size));

		/* Generate ray direction */
		glm::vec3 rayDir = glm::normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
		float3 ray_direction = { rayDir.x, rayDir.y, rayDir.z };
		float3 ray_origin = { camera.position.x, camera.position.y, camera.position.z };

		/* Launch ray */
		optixTrace(
			optixLaunchParams.traversable,
			ray_origin,
			ray_direction,
			0.0f, /* tMin */
			1e20f, /* tMax */
			0.0f, /* ray time */
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT, /* OPTIX_RAY_FLAG_NONE */
			SURFACE_RAY_TYPE, /* SBT offset */
			RAY_TYPE_COUNT, /* SBT stride */
			SURFACE_RAY_TYPE, /* miss SBT index */
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