#include "utils.cuh"

#include "../../common_enums.h"

namespace otx
{
	extern "C" __global__ void __miss__radiance()
	{
		PRD_Radiance& prd = *getPRD<PRD_Radiance>();

		/* Output variable */
		float3 result = make_float3(0.0f);

		switch (optixLaunchParams.backgroundMode)
		{
		case BACKGROUND_MODE_SOLID_COLOR:
		{
			result = optixLaunchParams.clearColor;
			break;
		}
		case BACKGROUND_MODE_GRADIENT:
		{
			float3 rayDir = optixGetWorldRayDirection();

			/* Dot rayDir with the up vector and use the result to interpolate between the bottom and top gradient colors */
			float t = max(dot(normalize(rayDir), make_float3(0.0f, 0.0f, 1.0f)), 0.0f);
			result = lerp(optixLaunchParams.gradientBottom, optixLaunchParams.gradientTop, t);
			break;
		}
		case BACKGROUND_MODE_TEXTURE:
		{
			float3 rayDir = optixGetWorldRayDirection();

			/* Convert the input ray direction to UV coordinates to access the background texture */
			float u = 0.5f * (1.0f + atan2(rayDir.x, rayDir.y) * M_1_PIf) + optixLaunchParams.backgroundRotation;
			float v = atan2(length(make_float2(rayDir.x, rayDir.y)), rayDir.z) * M_1_PIf;

			float4 tex = tex2D<float4>(optixLaunchParams.backgroundTexture, u, v);
			result = make_float3(tex.x, tex.y, tex.z);
			break;
		}
		}

		prd.throughput *= result;
		prd.done = true;
	}
}