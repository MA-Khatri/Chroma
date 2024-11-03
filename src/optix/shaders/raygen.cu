#include "utils.cuh"

namespace otx
{
	inline __device__ float2 GenerateScreenPosition(int ix, int iy, Random& random)
	{
		/* Normalized screen plane position in [0, 1]^2 with randomized sub-pixel position */
		return (make_float2((float)ix, (float)iy) + random.RandomSample2D()) / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);
	}


	inline __device__ void GenerateCameraRay(PRD_Radiance& prd, float2 screen)
	{
		/* Get the camera from launchParams */
		const auto& camera = optixLaunchParams.camera;

		switch (optixLaunchParams.camera.projectionMode)
		{
		case PROJECTION_MODE_PERSPECTIVE:
		{
			prd.origin = camera.position;
			prd.in_direction = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
			break;
		}

		case PROJECTION_MODE_ORTHOGRAPHIC:
		{
			prd.origin = camera.position + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical;
			prd.in_direction = camera.direction;
			break;
		}
		case PROJECTION_MODE_THIN_LENS:
		{
			float2 p = prd.random.RandomInUnitDisk();
			float3 orgOffset = (p.x * camera.defocusDiskU) + (p.y * camera.defocusDiskV);
			prd.origin = camera.position + orgOffset;
			prd.in_direction = normalize(camera.direction + ((screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical) - orgOffset);
			break;
		}
		}
	}


	__device__ void BSDFIntegrator(PRD_Radiance& prd, uint32_t u0, uint32_t u1)
	{
		/* Initial prd values -- origin, in_direction already set */
		prd.depth = 0;
		prd.done = false;
		prd.sbtData = nullptr;
		prd.throughput = make_float3(1.0f);
		prd.color = make_float3(0.0f);

		/* === Iterative path tracing loop === */
		while (true)
		{
			optixTrace(
				optixLaunchParams.traversable,
				prd.origin,
				prd.in_direction,
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
			prd.depth++;

			/* If the ray has terminated (e.g. hit a light / miss), end */
			if (prd.done)
			{
				prd.color += prd.throughput;
				break;
			}

			/* If max depth == 0, we use russian roulette to determine path termination */
			if (optixLaunchParams.maxDepth == 0)
			{
				/* 
				 * We do not start russian roulette path termination until after first 
				 * 3 bounces to make sure we can get at least some lighting... 
				 */
				if (prd.depth > 3)
				{
					/* Clamp russian roulette to 0.99f to prevent inf bounces for materials that do not absorb any light */
					float p = min(max(prd.throughput.x, max(prd.throughput.y, prd.throughput.z)), 0.99f);
					if (prd.random() > p)
					{
						break;
					}
					prd.throughput /= p;
				}
			}
			/* Terminate the random walk if we're at/past the max depth */
			else if (prd.depth >= optixLaunchParams.maxDepth)
			{
				prd.color += optixLaunchParams.cutoffColor * prd.throughput;
				break;
			}
		}
	}


	//__device__ void PathIntegrator(PRD_Radiance& prd, uint32_t u0, uint32_t u1)
	//{
	//	/* Initial prd values -- origin and direction are already set */
	//	prd.depth = 0;
	//	prd.done = false;
	//	prd.radiance = make_float3(1.0f);
	//	prd.totalRadiance = make_float3(0.0f);
	//	prd.bsdfPDF = 1.0f;
	//	prd.nLightPaths = 0;

	//	/* === Iterative path tracing loop === */
	//	float cbPDF = 1.0f; /* product pdf for bsdf sampling path */
	//	while (true)
	//	{
	//		/* Take a random walk step */
	//		optixTrace(
	//			optixLaunchParams.traversable,
	//			prd.origin,
	//			prd.direction,
	//			0.0f, /* tMin */
	//			1e20f, /* tMax */
	//			0.0f, /* ray time */
	//			OptixVisibilityMask(255),
	//			OPTIX_RAY_FLAG_DISABLE_ANYHIT,
	//			RAY_TYPE_RADIANCE, /* SBT offset */
	//			RAY_TYPE_COUNT, /* SBT stride */
	//			RAY_TYPE_RADIANCE, /* miss SBT index */
	//			u0, u1 /* packed pointer to our PRD */
	//		);
	//		prd.depth++;

	//		cbPDF *= prd.bsdfPDF;

	//		/* Sample a light(s) */
	//		float clPDF = 0.0f;
	//		float3 cLightRadiance = make_float3(0.0f);

	//		if (prd.depth < optixLaunchParams.maxDepth) /* Skip sampling light if we're past max depth */
	//			for (int i = 0; i < optixLaunchParams.lightSampleCount; i++)
	//			{
	//				/* Pick a light to sample... */
	//				// TODO

	//				/* For now we just pick a point on the surface of a single quad light */
	//				float r1 = prd.random();
	//				float r2 = prd.random();
	//				float3 lightSamplePosition = make_float3(r1 * 1.0f - 0.5f, r2 * 1.0f - 0.5f, 9.98f);
	//				float3 lightSampleDirection = lightSamplePosition - prd.origin;
	//				float3 lightNormalDirection = make_float3(0.0f, 0.0f, -1.0f);
	//				float3 normalizedLightSampleDirection = normalize(lightSampleDirection);
	//				float lightSampleLength = length(lightSampleDirection);

	//				/* Get info of chosen light */
	//				bool isDeltaLight = false;
	//				float lightArea = 1.0f;
	//				float3 lightRadiance = make_float3(50.0f);

	//				/* Only emit from front face of light */
	//				if (!isDeltaLight && dot(normalizedLightSampleDirection, lightNormalDirection) >= -RAY_EPS) continue;

	//				/* Light does not illuminate back faces */
	//				if (dot(normalizedLightSampleDirection, prd.basis.w) <= RAY_EPS) continue;

	//				/* Initialize a shadow ray... */
	//				PRD_Shadow shadowRay;
	//				shadowRay.radiance = make_float3(0.0f);
	//				shadowRay.reachedLight = false;

	//				/* Launch the shadow ray towards the selected light */
	//				uint32_t s0, s1;
	//				packPointer(&shadowRay, s0, s1);
	//				optixTrace(
	//					optixLaunchParams.traversable,
	//					prd.origin, /* I.e., last hit position of the primary ray path */
	//					normalizedLightSampleDirection,
	//					0.0f, /* prd.origin should already be offset */
	//					lightSampleLength - RAY_EPS,
	//					0.0f, /* ray time */
	//					OptixVisibilityMask(255),
	//					OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
	//					RAY_TYPE_SHADOW,
	//					RAY_TYPE_COUNT,
	//					RAY_TYPE_SHADOW,
	//					s0, s1
	//				);

	//				if (shadowRay.reachedLight)
	//				{
	//					/* Probability of light scattering in light sample direction */
	//					float scatteringPDF = 0.0f;
	//					switch (prd.shadowRayPDFMode)
	//					{
	//					case PDF_UNIT_COSINE_HEMISPHERE:
	//						scatteringPDF = CosineHemispherePDF(normalizedLightSampleDirection, prd.basis.w);
	//						break;
	//					case PDF_UNIT_HEMISPHERE:
	//						scatteringPDF = UnitHemispherePDF();
	//						break;
	//					case PDF_UNIT_SPHERE:
	//						scatteringPDF = UnitSpherePDF();
	//						break;
	//					case PDF_DELTA:
	//						scatteringPDF = DeltaPDF(prd.direction, normalizedLightSampleDirection);
	//						break;
	//					}

	//					/* Probability of sampling the light from this point */
	//					float lightPDF;
	//					if (isDeltaLight)
	//					{
	//						/* inverse square law (over sphere surface area) */
	//						lightPDF = 4.0f * M_PIf / (lightSampleLength * lightSampleLength);
	//					}
	//					else
	//					{
	//						/* inverse square law with light area and cosine (over hemisphere surface area) */
	//						lightPDF = 2.0f * M_PIf * lightArea * max(dot(normalizedLightSampleDirection, -lightNormalDirection), 0.0f) / (lightSampleLength * lightSampleLength);
	//					}

	//					float lPDF = cbPDF * scatteringPDF * lightPDF / float(optixLaunchParams.lightSampleCount);
	//					clPDF += lPDF;
	//					cLightRadiance += lightRadiance * lPDF;
	//					prd.nLightPaths += 1.0f / float(optixLaunchParams.lightSampleCount);
	//				}
	//			}
	//		prd.totalRadiance += prd.radiance * cLightRadiance / (cbPDF + clPDF);

	//		/* If the random walk has terminated (e.g. hit a light / miss), end */
	//		if (prd.done)
	//		{
	//			prd.totalRadiance += prd.radiance * cbPDF / (cbPDF + clPDF);
	//			prd.nLightPaths += 1.0f;
	//			break;
	//		}

	//		/* Terminate the random walk if we're at/past the max depth */
	//		if (prd.depth >= optixLaunchParams.maxDepth)
	//		{
	//			prd.totalRadiance += optixLaunchParams.cutoffColor * prd.radiance * cbPDF / (cbPDF + clPDF);
	//			prd.nLightPaths += 1.0f;
	//			break;
	//		}
	//	}

	//	prd.totalRadiance /= prd.nLightPaths;
	//}


	inline __device__ void Integrate(PRD_Radiance& prd, uint32_t u0, uint32_t u1)
	{
		switch (optixLaunchParams.integrator)
		{
		case INTEGRATOR_TYPE_BSDF_ONLY:
			BSDFIntegrator(prd, u0, u1);
			break;

		//case INTEGRATOR_TYPE_PATH:
		//	PathIntegrator(prd, u0, u1);
		//	break;
		}
	}


	/*
	 * The primary ray gen program that is called on Optix::Render()
	 */
	extern "C" __global__ void __raygen__renderFrame()
	{
		/* Get pixel position and framebuffer index */
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;
		const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

		/* Get the current frame's frameID -- i.e., which render call is this? */
		const int accumID = optixLaunchParams.frame.frameID;

		/* Initialize per-ray data */
		PRD_Radiance prd;

		/* Random seed is current frame count * frame size + current (1D) pixel position such that every pixel for every accumulated frame has a unique seed. */
		prd.random.Init(accumID * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + iy * optixLaunchParams.frame.size.x + ix, optixLaunchParams.sampler, optixLaunchParams.nStrata);

		/* The ints we store the PRD pointer in */
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		const int numPixelSamples = optixLaunchParams.frame.samples; /* N Pixel samples for this render call */
		float3 pixelColor = make_float3(0.0f); /* Accumulated color for all pixel samples for this call */
		float3 pixelNormal = make_float3(0.0f); /* Accumulated normals for all pixel samples for this call */
		float3 pixelAlbedo = make_float3(0.0f); /* Accumulated albedo for all pixel samples for this call */
		for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
		{
			/* Determine the screen sampling position and generate corresponding camera ray */
			float2 screen = GenerateScreenPosition(ix, iy, prd.random);
			GenerateCameraRay(prd, screen);

			/* Run the integrator for this sample -- result is stored in prd.color */
			Integrate(prd, u0, u1);

			/* Set NaNs to 0 */
			if (prd.color.x != prd.color.x) prd.color.x = 0.0f;
			if (prd.color.y != prd.color.y) prd.color.y = 0.0f;
			if (prd.color.z != prd.color.z) prd.color.z = 0.0f;

			pixelColor += prd.color;
			pixelNormal += prd.normal;
			pixelAlbedo += prd.albedo;
		}
		
		/* Determine average color for this call. Cap to prevent speckles (even though this breaks pbr condition) */
		const float cap = 100.0f;
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