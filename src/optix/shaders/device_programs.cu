#include <optix_device.h>
#include <cuda_runtime.h>

#include "../launch_params.h"

#include "utils.cuh"

namespace otx
{
	typedef PCG Random;

	/* Launch parameters in constant memory, filled in by Optix upon optixLaunch */
	extern "C" __constant__ LaunchParams optixLaunchParams;

	/* Per-ray data */
	struct PRD
	{
		Random random;
		float3 pixelColor;
		int depth = 0;
	};

	/* ===================== */
	/* === Radiance Rays === */
	/* ===================== */
	extern "C" __global__ void __closesthit__radiance()
	{
		const MeshSBTData& sbtData = *(const MeshSBTData*)optixGetSbtDataPointer();
		PRD& prd = *getPRD<PRD>();

		const int primID = optixGetPrimitiveIndex();
		const int3 index = sbtData.index[primID];
		float2 uv = optixGetTriangleBarycentrics();
		float3 rayDir = optixGetWorldRayDirection();

		/* === Compute normal === */
		/* Use shading normal if available, else use geometry normal */
		const float3& v0 = sbtData.position[index.x];
		const float3& v1 = sbtData.position[index.y];
		const float3& v2 = sbtData.position[index.z];
		float3 Ng = cross(v1 - v0, v2 - v0);
		float3 Ns = (sbtData.normal) ? InterpolateNormals(uv, sbtData.normal[index.x], sbtData.normal[index.y], sbtData.normal[index.z]) : Ng;

		/* Compute world-space normal and normalize */
		Ns = normalize(optixTransformNormalFromObjectToWorldSpace(Ns));

		/* Default diffuse color if no diffuse texture */
		float3 diffuseColor = *sbtData.color;

		/* === Sample texture(s) === */
		float2 tc = TexCoord(uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			diffuseColor = make_float3(tex.x, tex.y, tex.z);
		}

		/* === Trace diffuse ray === */
		if (prd.depth < 8)
		{
			/* Determine diffuse ray origin and reflection direction */
			OrthonormalBasis basis = OrthonormalBasis(Ns);
			float3 reflectDir = basis.Local(prd.random.RandomOnUnitCosineHemisphere());
			float3 reflectOrigin = HitPosition() + 1e-3f * Ns;


			/* Launch reflected ray */
			PRD reflectPRD;
			reflectPRD.random = prd.random;
			reflectPRD.pixelColor = make_float3(1.0f);
			reflectPRD.depth = prd.depth + 1;

			uint32_t u0, u1;
			packPointer(&reflectPRD, u0, u1);

			optixTrace(
				optixLaunchParams.traversable,
				reflectOrigin,
				reflectDir,
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

			/* Multiply ray colors together */
			prd.pixelColor = diffuseColor * reflectPRD.pixelColor;
		}
		else
		{
			prd.pixelColor = diffuseColor;
		}
		
		///* === Compute shadow === */
		//const float3 surfPosn = HitPosition();
		//const float3 lightPosn = make_float3(100.0f, 100.0f, 100.0f); /* Hard coded light position (for now) */
		//const float3 lightDir = lightPosn - surfPosn;

		///* Trace shadow ray*/
		//float3 lightVisibility = make_float3(0.5f);
		//uint32_t u0, u1;
		//packPointer(&lightVisibility, u0, u1);
		//optixTrace(
		//	optixLaunchParams.traversable,
		//	surfPosn + 1e-3f * Ns,
		//	lightDir,
		//	1e-3f, /* tmin */
		//	1.0f-1e-3f, /* tmax -- in terms of lightDir length */
		//	0.0f, /* ray time */
		//	OptixVisibilityMask(255),
		//	OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		//	SHADOW_RAY_TYPE, /* SBT offset */
		//	RAY_TYPE_COUNT, /* SBT stride */
		//	SHADOW_RAY_TYPE, /* missSBT index */
		//	u0, u1 /* packed pointer to our PRD */
		//);


		///* Calculate shading */
		//const float ambient = 0.2f;
		//const float diffuse = 0.5f;
		//const float specular = 0.1f;
		//const float exponent = 16.0f;

		//const float3 reflectDir = reflect(rayDir, Ns);
		//const float diffuseContrib = clamp(dot(-rayDir, Ns), 0.0f, 1.0f);
		//const float specularContrib = pow(max(dot(-rayDir, reflectDir), 0.0f), exponent);
		//const float lc = ambient + diffuse * diffuseContrib + specular * specularContrib;

		//prd.pixelColor *= (lightVisibility * lc);
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO
	}

	extern "C" __global__ void __miss__radiance()
	{
		PRD& prd = *getPRD<PRD>();
		//prd.pixelColor = optixLaunchParams.clearColor;
		prd.pixelColor = make_float3(1.0f);
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
		/* Get pixel position and framebuffer index */
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;
		const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

		/* Get the current frame's accumulation ID */
		const int accumID = optixLaunchParams.frame.accumID;

		/* Get the camera from launchParams */
		const auto& camera = optixLaunchParams.camera;

		/* Get the current pixel's accumulated color */
		float3 aclr = make_float3(0.0f);
		if (accumID > 0)
		{
			float r = optixLaunchParams.frame.accumBuffer[fbIndex * 3 + 0];
			float g = optixLaunchParams.frame.accumBuffer[fbIndex * 3 + 1];
			float b = optixLaunchParams.frame.accumBuffer[fbIndex * 3 + 2];
			aclr = make_float3(r, g, b);
		}

		/* Initialize per-ray data */
		PRD prd;
		/* Random seed is current frame count * frame size + current (1D) pixel position such that every pixel for every accumulated frame has a unique seed. */
		prd.random.Init(accumID * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + iy * optixLaunchParams.frame.size.x + ix);
		prd.pixelColor = make_float3(1.0f);

		/* The ints we store the PRD pointer in */
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		const int numPixelSamples = optixLaunchParams.frame.samples; /* Pixel samples per call to render */
		float3 pixelColor = make_float3(0.0f);
		for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
		{
			/* Normalized screen plane position in [0, 1]^2 with randomized sub-pixel position */
			const float2 screen = make_float2(ix + prd.random(), iy + prd.random()) / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);

			/* Ray origin and direction */
			float3 rayOrg, rayDir;

			if (optixLaunchParams.camera.projectionMode == 0) /* Camera::PERSPECTIVE */
			{
				rayOrg = camera.position;
				rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
			}
			else if (optixLaunchParams.camera.projectionMode == 1) /* Camera::ORTHOGRAPHIC */
			{

				rayOrg = camera.position + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical;
				rayDir = camera.direction;
			}

			/* Launch ray */
			optixTrace(
				optixLaunchParams.traversable,
				rayOrg,
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
			pixelColor += prd.pixelColor;
		}
		
		/* Determine average color for this call */
		const float cr = min(pixelColor.x / numPixelSamples, 1.0f);
		const float cg = min(pixelColor.y / numPixelSamples, 1.0f);
		const float cb = min(pixelColor.z / numPixelSamples, 1.0f);
		const float3 cclr = make_float3(cr, cg, cb);

		/* Determine the new accumulated color */
		float3 tclr = (cclr + accumID * aclr) / (accumID + 1);
		tclr = make_float3(min(tclr.x, 1.0f), min(tclr.y, 1.0f), min(tclr.z, 1.0f));

		/* Update the accumulated color buffer */
		optixLaunchParams.frame.accumBuffer[fbIndex * 3 + 0] = tclr.x;
		optixLaunchParams.frame.accumBuffer[fbIndex * 3 + 1] = tclr.y;
		optixLaunchParams.frame.accumBuffer[fbIndex * 3 + 2] = tclr.z;

		/* Convert accumulated color to ints */
		const int r = int(255.99f * tclr.x);
		const int g = int(255.99f * tclr.y);
		const int b = int(255.99f * tclr.z);

		/* Convert to 32-bit RGBA value, explicitly setting alpha to 0xff */
		const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

		/* Write to the frame buffer */
		optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
	}

} /* namespace otx */