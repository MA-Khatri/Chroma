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

			/* Dot rayDir with the up vector and use the result to interpolate between the bottom and top gradient colors */
			const float t = max(dot(normalize(rayDir), make_float3(0.0f, 0.0f, 1.0f)), 0.0f);
			result = lerp(optixLaunchParams.gradientBottom, optixLaunchParams.gradientTop, t);
		}
		else if (optixLaunchParams.backgroundMode == BackgroundMode::TEXTURE)
		{
			float3 rayDir = optixGetWorldRayDirection();

			/* Convert the input ray direction to UV coordinates to access the background texture */
			float u = 0.5f * (1.0f + atan2(rayDir.x, rayDir.y) * M_1_PIf);
			float v = atan2(length(make_float2(rayDir.x, rayDir.y)), rayDir.z) * M_1_PIf;

			float4 tex = tex2D<float4>(optixLaunchParams.backgroundTexture, u, v);
			result = make_float3(tex.x, tex.y, tex.z);
		}
		else
		{
			result = make_float3(0.0f);
		}

		prd.radiance *= result;
		prd.done = true;
	}
}