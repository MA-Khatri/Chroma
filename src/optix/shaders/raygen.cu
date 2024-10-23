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

		/* Get the current frame's frameID -- i.e., which render call is this? */
		const int accumID = optixLaunchParams.frame.frameID;

		/* Get the camera from launchParams */
		const auto& camera = optixLaunchParams.camera;

		/* Initialize per-ray data */
		PRD_Radiance prd;

		/* Random seed is current frame count * frame size + current (1D) pixel position such that every pixel for every accumulated frame has a unique seed. */
		prd.random.Init(accumID * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + iy * optixLaunchParams.frame.size.x + ix);

		/* The ints we store the PRD pointer in */
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		const int nps = optixLaunchParams.frame.samples;
		const int numPixelSamples = optixLaunchParams.stratifiedSampling ? nps * nps : nps; /* N Pixel samples for this render call */
		const float spd = 1.0f / float(nps); /* sub-pixel delta (spd) used for stratified sampling */
		float3 pixelColor = make_float3(0.0f); /* Accumulated color for all pixel samples for this call */
		float3 pixelNormal = make_float3(0.0f); /* Accumulated normals for all pixel samples for this call */
		float3 pixelAlbedo = make_float3(0.0f); /* Accumulated albedo for all pixel samples for this call */
		for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
		{
			/* Initial prd values for the random walk */
			prd.depth = 0;
			prd.done = false;
			prd.radiance = make_float3(1.0f);
			prd.totalRadiance = make_float3(0.0f);
			prd.nLightPaths = 0;
			prd.origin = make_float3(0.0f);
			prd.direction = make_float3(0.0f);

			/* Determine the screen sampling position */
			float2 screen;
			if (optixLaunchParams.stratifiedSampling)
			{
				/* Determine the sub pixel offset (spo) for this sampleID */
				float2 spo = make_float2(float(sampleID % nps), float(sampleID / nps));

				/* Determine posn within sub pixel (spp) */
				float2 spp = make_float2(prd.random(), prd.random());

				/* Normalized screen plane position in [0, 1]^2 with stratified random sub-pixel position */
				screen = (make_float2(ix, iy) + (spo + spp) * spd) / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);
			}
			else
			{
				/* Normalized screen plane position in [0, 1]^2 with randomized sub-pixel position */
				screen = make_float2(ix + prd.random(), iy + prd.random()) / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);
			}

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
				/* Take a random walk step */
				optixTrace(
					optixLaunchParams.traversable,
					rayOrg,
					rayDir,
					0.0f, /* tMin */
					1e20f, /* tMax */
					0.0f, /* ray time */
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_DISABLE_ANYHIT,
					RAY_TYPE_RADIANCE, /* SBT offset */
					RAY_TYPE_COUNT, /* SBT stride */
					RAY_TYPE_RADIANCE, /* miss SBT index */
					u0, u1 /* packed pointer to our PRD */
				);
				
				/* Set the ray orign and direction for the next segment of the random walk */
				rayOrg = prd.origin;
				rayDir = prd.direction;
				prd.depth++;

				/* If the random walk has terminated (e.g. hit a light / miss), end */
				if (prd.done)
				{
					prd.totalRadiance += prd.radiance; /* Add the primary ray path's radiance */
					prd.nLightPaths++;
					break;
				}

				/* Terminate the random walk if we're at/past the max depth */
				if (prd.depth >= optixLaunchParams.maxDepth)
				{
					prd.radiance *= optixLaunchParams.cutoffColor;
					prd.totalRadiance += prd.radiance;
					prd.nLightPaths++;
					break;
				}
			}

			pixelColor += prd.totalRadiance / float(prd.nLightPaths);
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