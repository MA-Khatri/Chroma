#include "utils.cuh"

namespace otx
{
	__forceinline__ __device__ float2 GenerateScreenPosition(int ix, int iy, Random& random)
	{
		/* Normalized screen plane position in [0, 1]^2 with randomized sub-pixel position */
		return (make_float2((float)ix, (float)iy) + random.RandomSample2D()) / make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);
	}


	__forceinline__ __device__ void GenerateCameraRay(PRD_Radiance& prd, float2 screen)
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


	__forceinline__ __device__ void BSDFIntegrator(PRD_Radiance& prd, uint32_t u0, uint32_t u1)
	{
		/* Initial prd values -- origin, in_direction already set */
		prd.depth = 0;
		prd.done = false;
		prd.throughput = make_float3(1.0f);
		prd.pdf = 1.0f;
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
					float p = min(prd.pdf, 0.99f);
					if (prd.random() > p)
					{
						break;
					}
					prd.pdf /= p;
				}
			}
			/* Terminate the random walk if we're at/past the max depth */
			else if (prd.depth >= optixLaunchParams.maxDepth) break;
		}
	}


	__forceinline__ __device__ void PathIntegrator(PRD_Radiance& prd, uint32_t u0, uint32_t u1)
	{
		/* Initial prd values -- origin, in_direction already set */
		prd.depth = 0;
		prd.done = false;
		prd.throughput = make_float3(1.0f);
		prd.pdf = 1.0f;
		prd.color = make_float3(0.0f);

		/* === Iterative path tracing loop === */
		while (true)
		{
			/* 
			 * Use RR to decide if we sample a light or sample the BSDF.
			 * Note: We must hit at least one surface before we start sampling the lights, 
			 * so we always start with a bsdf sample.
			 */
			float rr = prd.random();
			if (rr < 0.5f || prd.depth == 0)
			{ /* === Sample the BSDF === */

				/* Shoot a ray according to the (previously set) bsdf sample origin and direction */
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

				/* Account for choosing to sample the bsdf instead of a light */
				prd.pdf /= prd.depth == 0 ? 1.0f : 2.0f;

				/* If the ray has terminated (e.g. hit a light / miss), end */
				if (prd.done)
				{
					/* Note: We do not need to use the power heuristic here since the shadowRay.pdf will always be 0.0f! */
					prd.color += prd.throughput;
					break;
				}

				/* If max depth == 0, we use russian roulette to determine path termination */
				if (optixLaunchParams.maxDepth == 0)
				{
					/*
					 * We do not start russian roulette path termination until after first
					 * 3 bounces to make sure we can get at least some indirect lighting...
					 */
					if (prd.depth > 3)
					{
						/* Clamp russian roulette to 0.99f to prevent inf bounces for materials that do not absorb any light */
						float p = min(prd.pdf, 0.99f);
						if (prd.random() > p)
						{
							break;
						}
						prd.pdf /= p;
					}
				}
				/* Not using RR, terminate the random walk if we're at/past the max depth */
				else if (prd.depth >= optixLaunchParams.maxDepth) break;
			}
			else
			{ /* === Sample a light === */

				/* Initialize a shadow ray... */
				PRD_Shadow shadowRay;
				shadowRay.throughput = make_float3(0.0f);
				shadowRay.pdf = 0.0f;
				shadowRay.reached_light = false;


				int nLights = 2; /* REPLACE THIS LATER WITH A LAUNCH PARAM -- Currently, 1 light + background */

				rr = prd.random();

				if (rr < 1.0f / (float)nLights)
				{ /* Importance sample the background */
					float3 lightSampleDirection = prd.random.RandomOnUnitSphere();

					/* Probability of sampling the light from this point -- background covers whole hemisphere */
					shadowRay.pdf = 1.0f;

					/* Account for the probability of choosing this light and for choosing to sample a light instead of the bsdf */
					shadowRay.pdf /= (float)nLights * 2.0f;

					/* Probability of light scattering in light sample direction */
					float scatteringPDF = optixDirectCall<float, PRD_Radiance&, float3>(prd.PDF, prd, lightSampleDirection);
					if (scatteringPDF > 0.0f) shadowRay.pdf /= scatteringPDF;
					else shadowRay.pdf = 0.0f;

					/* Only trace the actual ray if the pdf is greater than 0.0f */
					if (shadowRay.pdf > 0.0f)
					{
						/* Launch the shadow ray towards the selected light */
						uint32_t s0, s1;
						packPointer(&shadowRay, s0, s1);
						optixTrace(
							optixLaunchParams.traversable,
							prd.origin, /* I.e., last hit position of the primary ray path */
							lightSampleDirection,
							0.0f, /* prd.origin should already be offset */
							1e-20f,
							0.0f, /* ray time */
							OptixVisibilityMask(255),
							OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
							RAY_TYPE_SHADOW,
							RAY_TYPE_COUNT,
							RAY_TYPE_SHADOW,
							s0, s1
						);

						float3 lightRadiance = optixDirectCall<float3, float3>(CALLABLE_SAMPLE_BACKGROUND, lightSampleDirection);

						if (shadowRay.reached_light)
						{
							shadowRay.throughput = lightRadiance / shadowRay.pdf;
							prd.color += powerHeuristic(shadowRay.pdf, prd.pdf) * prd.throughput * shadowRay.throughput;
							break;
						}
					}
				}
				else
				{/* Pick a light to sample... */
					// TODO

					/* For now we just pick a point on the surface of a single quad light */
					float2 rand2d = prd.random.RandomSample2D();
					float3 lightSamplePosition = make_float3(rand2d.x * 1.0f - 0.5f, rand2d.y * 1.0f - 0.5f, 9.98f);
					float3 lightSampleDirection = lightSamplePosition - prd.origin;
					float3 lightNormalDirection = make_float3(0.0f, 0.0f, -1.0f);
					float3 normalizedLightSampleDirection = normalize(lightSampleDirection);
					float lightSampleLength = length(lightSampleDirection);

					/* Get info about chosen light */
					bool isDeltaLight = false;
					float lightArea = 1.0f;
					float3 lightRadiance = make_float3(50.0f);


					/* Probability of sampling the light from this point */
					if (isDeltaLight)
					{
						shadowRay.pdf = 1.0f; // TODO
					}
					else
					{
						float cosTheta = max(dot(normalizedLightSampleDirection, -lightNormalDirection), 0.0f);
						shadowRay.pdf = cosTheta > 0.0f ? (lightSampleLength * lightSampleLength) / (lightArea * cosTheta) : 0.0f;
					}

					/* Account for the probability of choosing this light and for choosing to sample a light instead of the bsdf */
					shadowRay.pdf /= (float)nLights * 2.0f;

					/* Probability of light scattering in light sample direction */
					float scatteringPDF = optixDirectCall<float, PRD_Radiance&, float3>(prd.PDF, prd, normalizedLightSampleDirection);
					if (scatteringPDF > 0.0f) shadowRay.pdf /= scatteringPDF;
					else shadowRay.pdf = 0.0f;

					/* Only trace the actual ray if the pdf is greater than 0.0f */
					if (shadowRay.pdf > 0.0f)
					{
						/* Launch the shadow ray towards the selected light */
						uint32_t s0, s1;
						packPointer(&shadowRay, s0, s1);
						optixTrace(
							optixLaunchParams.traversable,
							prd.origin, /* I.e., last hit position of the primary ray path */
							normalizedLightSampleDirection,
							0.0f, /* prd.origin should already be offset */
							lightSampleLength - RAY_EPS,
							0.0f, /* ray time */
							OptixVisibilityMask(255),
							OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
							RAY_TYPE_SHADOW,
							RAY_TYPE_COUNT,
							RAY_TYPE_SHADOW,
							s0, s1
						);

						if (shadowRay.reached_light)
						{
							shadowRay.throughput = lightRadiance / shadowRay.pdf;
							prd.color += powerHeuristic(shadowRay.pdf, prd.pdf) * prd.throughput * shadowRay.throughput;
							break;
						}
					}
				}
			}
		}
	}


	__forceinline__ __device__ void Integrate(PRD_Radiance& prd, uint32_t u0, uint32_t u1)
	{
		switch (optixLaunchParams.integrator)
		{
		case INTEGRATOR_TYPE_BSDF_ONLY:
		{
			BSDFIntegrator(prd, u0, u1);
			break;
		}
		case INTEGRATOR_TYPE_PATH:
		{
			PathIntegrator(prd, u0, u1);
			break;
		}
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