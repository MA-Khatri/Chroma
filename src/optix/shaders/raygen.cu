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
		PRD_radiance prd;
		prd.depth = 0;
		prd.done = false;
		prd.radiance = make_float3(1.0f);
		prd.origin = make_float3(0.0f);
		prd.direction = make_float3(0.0f);

		/* Random seed is current frame count * frame size + current (1D) pixel position such that every pixel for every accumulated frame has a unique seed. */
		prd.random.Init(accumID * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + iy * optixLaunchParams.frame.size.x + ix);

		/* The ints we store the PRD pointer in */
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		const int numPixelSamples = optixLaunchParams.frame.samples; /* Pixel samples per call to render */
		float3 pixelColor = make_float3(0.0f); /* Accumulated color for all pixel samples */
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
		}
		
		/* Determine average color for this call. Cap to prevent speckles (even though this breaks pbr condition) */
		const float cap = 5.0f;
		const float cr = min(pixelColor.x / numPixelSamples, cap);
		const float cg = min(pixelColor.y / numPixelSamples, cap);
		const float cb = min(pixelColor.z / numPixelSamples, cap);
		//const float cr = pixelColor.x / numPixelSamples;
		//const float cg = pixelColor.y / numPixelSamples;
		//const float cb = pixelColor.z / numPixelSamples;
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