#include "utils.cuh"

namespace otx
{
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

		/* Initialize per-ray data */
		PRD_radiance prd;

		/* Random seed is current frame count * frame size + current (1D) pixel position such that every pixel for every accumulated frame has a unique seed. */
		prd.random.Init(accumID * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + iy * optixLaunchParams.frame.size.x + ix);

		/* The ints we store the PRD pointer in */
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		const int numPixelSamples = optixLaunchParams.frame.samples; /* Pixel samples per call to render */
		float3 pixelColor = make_float3(0.0f); /* Accumulated color for all pixel samples */
		float3 pixelNormal = make_float3(0.0f); /* Accumulated normals for all pixel samples */
		float3 pixelAlbedo = make_float3(0.0f); /* Accumulated albedo for all pixel samples */
		for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
		{
			/* Initial prd values */
			prd.depth = 0;
			prd.done = false;
			prd.radiance = make_float3(1.0f);
			prd.origin = make_float3(0.0f);
			prd.direction = make_float3(0.0f);

			/* Normalized screen plane position in [0, 1]^2 with randomized sub-pixel position */
			const float2 screen = make_float2(ix + prd.random(), iy + prd.random()) / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);

			/* Ray origin and direction */
			float3 rayOrg, rayDir;

			if (optixLaunchParams.camera.projectionMode == PROJECTION_MODE_PERSPECTIVE)
			{
				rayOrg = camera.position;
				rayDir = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
			}
			else if (optixLaunchParams.camera.projectionMode == PROJECTION_MODE_ORTHOGRAPHIC)
			{
				rayOrg = camera.position + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical;
				rayDir = camera.direction;
			}
			else if (optixLaunchParams.camera.projectionMode == PROJECTION_MODE_THIN_LENS)
			{
				float2 p = prd.random.RandomInUnitDisk();
				float3 orgOffset = (p.x * camera.defocusDiskU) + (p.y * camera.defocusDiskV);
				rayOrg = camera.position + orgOffset;
				rayDir = normalize(camera.direction + ((screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical) - orgOffset);
			}

			/* Iterative (non-recursive) render loop */
			while (true)
			{
				if (prd.depth >=  optixLaunchParams.maxDepth)
				{
					prd.radiance *= optixLaunchParams.cutoffColor;
					break;
				}

				optixTrace(
					optixLaunchParams.traversable,
					rayOrg,
					rayDir,
					0.0f, /* tMin */
					1e20f, /* tMax */
					0.0f, /* ray time */
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT, /* OPTIX_RAY_FLAG_NONE */
					RAY_TYPE_RADIANCE, /* SBT offset */
					RAY_TYPE_COUNT, /* SBT stride */
					RAY_TYPE_RADIANCE, /* miss SBT index */
					u0, u1 /* packed pointer to our PRD */
				);

				if (prd.done) break;

				/* Update ray data for next ray path segment */
				rayOrg = prd.origin;
				rayDir = prd.direction;

				prd.depth++;
			}

			pixelColor += prd.radiance;
			pixelNormal += prd.normal;
			pixelAlbedo += prd.albedo;
		}
		
		/* Determine average color for this call. Cap to prevent speckles (even though this breaks pbr condition) */
		const float cap = 1000.0f;
		const float cr = min(pixelColor.x / numPixelSamples, cap);
		const float cg = min(pixelColor.y / numPixelSamples, cap);
		const float cb = min(pixelColor.z / numPixelSamples, cap);
		const float4 ccolor = make_float4(cr, cg, cb, 1.0f);

		/* Determine the average albedo and normal for this call */
		pixelAlbedo = pixelAlbedo / numPixelSamples;
		const float4 albedo = make_float4(pixelAlbedo.x, pixelAlbedo.y, pixelAlbedo.z, 1.0f);

		pixelNormal = pixelNormal / numPixelSamples;
		const float4 normal = make_float4(pixelNormal.x, pixelNormal.y, pixelNormal.z, 1.0f);

		/* Get the current pixel's previously accumulated color, albedo, normal */
		float4 acolor = make_float4(0.0f);
		float4 aalbedo = make_float4(0.0f);
		float4 anormal = make_float4(0.0f);
		if (accumID > 0)
		{
			acolor = optixLaunchParams.frame.colorBuffer[fbIndex];
			aalbedo = optixLaunchParams.frame.albedoBuffer[fbIndex];
			anormal = optixLaunchParams.frame.normalBuffer[fbIndex];
		}

		/* Determine the new accumulated color, albedo, and normal */
		float4 tcolor = (ccolor + accumID * acolor) / (accumID + 1);
		tcolor = make_float4(min(tcolor.x, 1.0f), min(tcolor.y, 1.0f), min(tcolor.z, 1.0f), 1.0f);

		float4 talbedo = (albedo + accumID * aalbedo) / (accumID + 1);
		float4 tnormal = (normal + accumID * anormal) / (accumID + 1);

		/* Update the buffers */
		optixLaunchParams.frame.colorBuffer[fbIndex] = tcolor;
		optixLaunchParams.frame.albedoBuffer[fbIndex] = talbedo;
		optixLaunchParams.frame.normalBuffer[fbIndex] = tnormal;
	}

} /* namespace otx */