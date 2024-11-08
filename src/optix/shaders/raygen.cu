#include "utils.cuh"

namespace otx
{
	/* ================================ */
	/* === Ray Generation Functions === */
	/* ================================ */

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


	/* =================== */
	/* === Integrators === */
	/* =================== */

	/* === Path Integrator and related helpers === */
	/*
	 * The version of the path integrator that was (finally) working correctly was based on this stack exchange post:
	 * https://computergraphics.stackexchange.com/questions/5152/progressive-path-tracing-with-explicit-light-sampling
	 */

	__forceinline__ __device__ float3 CalculateDirectLightSamplePDF(PRD_Radiance& prd, PRD_Shadow& shadowRay, int l, float3& lightSampleDirection, float& distance)
	{
		/* Get the corresponding light */
		MISLight light;
		if (l == 0) light.type = LIGHT_TYPE_BACKGROUND;
		else light = *(optixLaunchParams.lights + (l - 1) * sizeof(MISLight));

		float3 lightRadiance = make_float3(0.0f);

		switch (light.type)
		{
		case LIGHT_TYPE_BACKGROUND:
		{
			/* Background sample can be anywhere on the unit sphere */
			lightSampleDirection = prd.random.RandomOnUnitSphere();
			distance = 1e20f;

			/* Probability of sampling this direction of the background */
			float cosTheta = max(dot(lightSampleDirection, make_float3(0.0f, 0.0f, 1.0f)), 0.0f);
			shadowRay.pdf = cosTheta > 0.0f ? 1.0f / cosTheta : 0.0f;

			lightRadiance = optixDirectCall<float3, float3>(CALLABLE_SAMPLE_BACKGROUND, lightSampleDirection);

			break;
		}
		case LIGHT_TYPE_AREA:
		{
			/* Sample a point on the triangle */
			float3 a = light.p1 - light.p0;
			float3 b = light.p2 - light.p0;

			float2 uv = prd.random.RandomSample2D();
			if (uv.x + uv.y > 1.0f) uv = make_float2(1.0f - uv.x, 1.0f - uv.y);

			float3 lightSamplePosition = light.p0 + a * uv.x + b * uv.y;
			float3 lsd = lightSamplePosition - prd.origin;
			float distance2 = dot(lsd, lsd);
			distance = sqrtf(distance2);
			lightSampleDirection = lsd / distance;

			/* Probability of sampling this point on the triangle */
			float3 lightNormal = InterpolateNormals(uv, light.n0, light.n1, light.n2);
			float cosTheta = max(dot(lightSampleDirection, -lightNormal), 0.0f);
			shadowRay.pdf = cosTheta > 0.0f ? distance2 / (light.area * cosTheta) : 0.0f;

			lightRadiance = light.emissionColor;

			break;
		}
		case LIGHT_TYPE_DELTA:
		{
			//TODO
			break;
		}
		}

		/* Subtract a small eps to prevent counting the intersection with the light surface itself */
		distance -= RAY_EPS;

		return lightRadiance;
	}


	__forceinline__ __device__ float3 CalculateBSDFLightSamplePDF(PRD_Radiance& prd, PRD_Shadow& shadowRay, int l, float3 lightSampleDirection, float& distance)
	{
		/* Note: The PDFs here will be the inverse of the corresponding PDFs in DirectLightSamplePDF */

		/* Get the corresponding light */
		MISLight light;
		if (l == 0) light.type = LIGHT_TYPE_BACKGROUND;
		else light = *(optixLaunchParams.lights + (l - 1) * sizeof(MISLight));

		float3 lightRadiance = make_float3(0.0f);

		switch (light.type)
		{
		case LIGHT_TYPE_BACKGROUND:
		{
			distance = 1e20f;

			/* Probability of sampling this direction of the background */
			float cosTheta = max(dot(lightSampleDirection, make_float3(0.0f, 0.0f, 1.0f)), 0.0f);
			shadowRay.pdf = cosTheta;

			lightRadiance = optixDirectCall<float3, float3>(CALLABLE_SAMPLE_BACKGROUND, lightSampleDirection);

			break;
		}
		case LIGHT_TYPE_AREA:
		{
			/* Trace the ray and see if it will intersect this triangle */
			/* https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm */
			float3 e1 = light.p1 - light.p0;
			float3 e2 = light.p2 - light.p0;
			float3 rce2 = cross(lightSampleDirection, e2);
			float det = dot(e1, rce2);

			if (det > -RAY_EPS && det < RAY_EPS)
			{ /* Ray is parallel */
				shadowRay.pdf = 0.0f;
				return lightRadiance;
			}

			float inv_det = 1.0f / det;
			float3 s = prd.origin - light.p0;
			float u = inv_det * dot(s, rce2);
			if (u < 0.0f || u > 1.0f)
			{ /* Out of bounds of triangle */
				shadowRay.pdf = 0.0f;
				return lightRadiance;
			}

			float3 sce1 = cross(s, e1);
			float v = inv_det * dot(lightSampleDirection, sce1);
			if (v < 0.0f || u + v > 1.0f)
			{ /* Out of bounds of triangle */
				shadowRay.pdf = 0.0f;
				return lightRadiance;
			}

			distance = inv_det * dot(e2, sce1);
			
			if (distance < RAY_EPS)
			{
				shadowRay.pdf = 0.0f;
				return lightRadiance;
			}

			/* Probability of sampling this point on the triangle */
			float3 lightNormal = InterpolateNormals(make_float2(u, v), light.n0, light.n1, light.n2);
			float cosTheta = max(dot(lightSampleDirection, -lightNormal), 0.0f);
			shadowRay.pdf = (light.area * cosTheta) / (distance * distance);

			lightRadiance = light.emissionColor;

			break;
		}
		case LIGHT_TYPE_DELTA:
		{
			//TODO
			break;
		}
		}

		/* Subtract a small eps to prevent counting the intersection with the light surface itself */
		distance -= RAY_EPS;

		return lightRadiance;
	}


	__forceinline__ __device__ float3 ImportanceSampleLight(PRD_Radiance& prd)
	{
		/* Initializing... */
		float3 directLighting = make_float3(0.0f);

		float3 bsdf;
		float scatteringPDF;

		float3 lightSampleDirection;
		float distance;

		/* Initialize a shadow ray... */
		PRD_Shadow shadowRay;
		shadowRay.throughput = make_float3(0.0f);
		shadowRay.pdf = 0.0f;
		shadowRay.reached_light = false;


		/* We add 1 light for the back ground... */
		int nLights = optixLaunchParams.nLights + 1;

		/* Choose a light to sample -- we can later use more advanced methods such as choosing based on light power */
		int l = (int)((float)nLights * prd.random());

		/* To speed up the frame rate, we randomly choose whether to sample the light directly or the via the bsdf */
		float lsr = optixLaunchParams.lightSampleRate;

		/* ============================= */
		/* === DIRECT LIGHT SAMPLING === */
		/* ============================= */

		/* Only sample the light directly if this was not a specular bounce */
		if (!prd.specular && prd.random() < lsr)
		{
			/*
			 * Calculate the PDF of sampling the chosen light (stored in shadowRay.pdf),
			 * the radiance of the light, and the distance to the chosen sample point
			 */
			float3 lightRadiance = CalculateDirectLightSamplePDF(prd, shadowRay, l, lightSampleDirection, distance);

			if (shadowRay.pdf > 0.0f)
			{
				/* Evaluate the BSDF for the chosen light sample direction */
				bsdf = optixDirectCall<float3, PRD_Radiance&, float3, float3>(prd.Eval, prd, lightSampleDirection, prd.out_direction);

				/* Probability of light scattering in the chosen light sample direction */
				scatteringPDF = optixDirectCall<float, PRD_Radiance&, float3>(prd.PDF, prd, lightSampleDirection);

				if (scatteringPDF > 0.0f && (bsdf.x > 0.0f || bsdf.y > 0.0f || bsdf.z > 0.0f))
				{
					/* Launch the shadow ray towards the selected light */
					uint32_t s0, s1;
					packPointer(&shadowRay, s0, s1);
					optixTrace(
						optixLaunchParams.traversable,
						prd.origin, /* I.e., last hit position of the primary ray path */
						lightSampleDirection,
						0.0f, /* prd.origin should already be offset */
						distance,
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
						directLighting += powerHeuristic(shadowRay.pdf, scatteringPDF) * lightRadiance * bsdf / shadowRay.pdf;
					}
				}
			}
		}
		else
		{
			/* =========================== */
			/* === BSDF LIGHT SAMPLING === */
			/* =========================== */

			/* Generate a sample direction from the bsdf */
			lightSampleDirection = optixDirectCall<float3, PRD_Radiance&>(prd.Sample, prd);

			/* Evaluate the bsdf for the chosen direction */
			bsdf = optixDirectCall<float3, PRD_Radiance&, float3, float3>(prd.Eval, prd, lightSampleDirection, prd.out_direction);

			/* Evaluate the pdf for the chosen direction */
			scatteringPDF = optixDirectCall<float, PRD_Radiance&, float3>(prd.PDF, prd, lightSampleDirection);
		
			if (scatteringPDF > 0.0f && (bsdf.x > 0.0f || bsdf.y > 0.0f || bsdf.z > 0.0f))
			{
				float3 lightRadiance = CalculateBSDFLightSamplePDF(prd, shadowRay, l, lightSampleDirection, distance);

				/* No bsdf sample if pdf is leq 0 */
				if (shadowRay.pdf <= 0.0f) return (float)nLights * directLighting;

				/* Launch the shadow ray towards the selected light */
				uint32_t s0, s1;
				packPointer(&shadowRay, s0, s1);
				optixTrace(
					optixLaunchParams.traversable,
					prd.origin, /* I.e., last hit position of the primary ray path */
					lightSampleDirection,
					0.0f, /* prd.origin should already be offset */
					distance,
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
					directLighting += powerHeuristic(scatteringPDF, shadowRay.pdf) * bsdf * lightRadiance / scatteringPDF;
				}
			}
		}

		/* We multiply by nLights to compensate for choosing this light */
		return (float)nLights * directLighting;
	}


	__forceinline__ __device__ void PathIntegrator(PRD_Radiance& prd)
	{
		/* Initial prd values -- origin, in_direction already set */
		prd.depth = 0;
		prd.done = false;
		prd.throughput = make_float3(1.0f);
		prd.pdf = 1.0f;
		prd.color = make_float3(0.0f);
		prd.Sample = CALLABLE_COUNT; /* I.e., invalid */

		uint32_t u0, u1;
		packPointer(&prd, u0, u1);

		/* We keep track of whether the previous hit was specular */
		bool previousHitSpecular = false;

		/* === Iterative path tracing loop === */
		while (true)
		{
			/* Trace the primary ray */
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

			/* If the ray has terminated (e.g., hit background), end */
			if (prd.done)
			{
				/* Note: Throughput already has already been multiplied by background color */
				prd.color += prd.throughput;
				break;
			}


			/* If this is the first bounce and we hit a light or if we just had a specular hit, we add light emission */
			if ((prd.depth == 1 || previousHitSpecular) && prd.Sample == CALLABLE_DIFFUSE_LIGHT_SAMPLE)
			{
				prd.color += prd.throughput;
			}

			/* Now we can update with the current hit's specular bool */
			previousHitSpecular = prd.specular;

			/* Importance sample the lights */
			prd.color += prd.throughput * ImportanceSampleLight(prd);

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
	}


	__forceinline__ __device__ void Integrate(PRD_Radiance& prd)
	{
		switch (optixLaunchParams.integrator)
		{
		case INTEGRATOR_TYPE_PATH:
		{
			PathIntegrator(prd);
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
			Integrate(prd);

			/* Set NaNs to 0 */
			if (prd.color.x != prd.color.x) prd.color.x = 0.0f;
			if (prd.color.y != prd.color.y) prd.color.y = 0.0f;
			if (prd.color.z != prd.color.z) prd.color.z = 0.0f;

			pixelColor += prd.color;
			pixelNormal += prd.normal;
			pixelAlbedo += prd.albedo;
		}

		/* Determine average color for this call. Cap to prevent speckles (even though this breaks pbr condition) */
		const float cap = 1e2f;
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