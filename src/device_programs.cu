#include <optix_device.h>

#include "launch_params.h" /* Also includes glm */

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

	/* Compute position of ray hit using barycentric coords */
	extern "C" __device__ glm::vec3 HitPosition(const float2& uv, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
	{
		return (1.0f - uv.x - uv.y) * p0 + uv.x * p1 + uv.y * p2;
	}
	
	/* Compute interpolated normal using barycentric coords */
	extern "C" __device__ glm::vec3 InterpolateNormals(const float2& uv, const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2)
	{
		return n0 + uv.x * (n1 - n0) + uv.y * (n2 - n0);
	}

	/* Compute texture coordinate using barycentric coords */
	extern "C" __device__ glm::vec2 TexCoord(const float2& uv, const glm::vec2& v0, const glm::vec2& v1, const glm::vec2& v2)
	{
		return (1.0f - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
	}

	extern "C" __device__ float3 ToFloat3(const glm::vec3& v)
	{
		return { v.x, v.y, v.z };
	}

	extern "C" __device__ glm::vec3 ToVec3(const float3& v)
	{
		return glm::vec3(v.x, v.y, v.z);
	}

	/* ===================== */
	/* === Radiance Rays === */
	/* ===================== */
	extern "C" __global__ void __closesthit__radiance()
	{
		const MeshSBTData& sbtData = *(const MeshSBTData*)optixGetSbtDataPointer();

		/* === Compute normal === */
		const int primID = optixGetPrimitiveIndex();
		const glm::ivec3 index = sbtData.index[primID];

		const glm::vec3& n0 = sbtData.normal[index.x];
		const glm::vec3& n1 = sbtData.normal[index.y];
		const glm::vec3& n2 = sbtData.normal[index.z];
		float2 uv = optixGetTriangleBarycentrics();
		glm::vec3 iN = InterpolateNormals(uv, n0, n1, n2);

		/* We need to clamp each element individually or the compiler will complain */
		glm::vec3 clampedNormals = glm::vec3(glm::clamp(iN.x, 0.0f, 1.0f), glm::clamp(iN.y, 0.0f, 1.0f), glm::clamp(iN.z, 0.0f, 1.0f));

		glm::vec3 diffuseColor = clampedNormals;

		/* === Sample texture(s) === */
		glm::vec2 tc = TexCoord(uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			diffuseColor = glm::vec3(tex.x, tex.y, tex.z);
		}

		/* === Compute shadow === */
		const glm::vec3 surfPosn = HitPosition(uv, sbtData.position[index.x], sbtData.position[index.y], sbtData.position[index.z]);
		const glm::vec3 lightPosn = glm::vec3(0.0f, 0.0f, 100.0f); /* Hard coded light position (for now) */
		const glm::vec3 lightDir = lightPosn - surfPosn;

		/* Trace shadow ray*/
		glm::vec3 lightVisibility = glm::vec3(0.5f);
		uint32_t u0, u1;
		packPointer(&lightVisibility, u0, u1);
		optixTrace(
			optixLaunchParams.traversable,
			ToFloat3(surfPosn + 1e-3f * iN),
			ToFloat3(-lightDir),
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

		/* === Set data === */
		glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
		prd = diffuseColor * lightVisibility;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO
	}

	extern "C" __global__ void __miss__radiance()
	{
		glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
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
		glm::vec3& prd = *(glm::vec3*)getPRD<glm::vec3>();
		prd = glm::vec3(1.0f);
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
		const glm::vec2 screen = glm::vec2(ix + 0.5f, iy + 0.5f) / glm::vec2(optixLaunchParams.frame.size);

		/* Generate ray direction */
		glm::vec3 rayDir = glm::normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);

		/* Launch ray */
		optixTrace(
			optixLaunchParams.traversable,
			ToFloat3(camera.position),
			ToFloat3(rayDir),
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