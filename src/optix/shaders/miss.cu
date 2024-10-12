#include "utils.cuh"

#include "../../background_mode.h"

namespace otx
{
	extern "C" __global__ void __miss__radiance()
	{
		PRD_radiance& prd = *getPRD<PRD_radiance>();

		float3 result;

		if (optixLaunchParams.backgroundMode == BackgroundMode::SOLID_COLOR)
		{
			result = optixLaunchParams.clearColor;
		}
		else if (optixLaunchParams.backgroundMode == BackgroundMode::GRADIENT)
		{
			float3 rayDir = optixGetWorldRayDirection();
			const float t = max(dot(normalize(rayDir), make_float3(0.0f, 0.0f, 1.0f)), 0.0f);
			result = lerp(optixLaunchParams.gradientBottom, optixLaunchParams.gradientTop, t);
		}
		else if (optixLaunchParams.backgroundMode == BackgroundMode::TEXTURE)
		{
			// TODO
		}
		else
		{
			result = make_float3(0.0f);
		}

		prd.radiance *= result;
		prd.done = true;
	}
}