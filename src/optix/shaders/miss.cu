#include "utils.cuh"

#include "../../common_enums.h"

namespace otx
{
	extern "C" __global__ void __miss__radiance()
	{
		PRD_Radiance& prd = *getPRD<PRD_Radiance>();

		/* Output variable */
		float3 result = make_float3(0.0f);

		/* Initialize variables used in switch cases */
		float3 rayDir = make_float3(0.0f);
		float t = 0.0f;
		float u = 0.0f;
		float v = 0.0f;
		float4 tex = make_float4(0.0f);

		switch (optixLaunchParams.backgroundMode)
		{
		case BACKGROUND_MODE_SOLID_COLOR:
			result = optixLaunchParams.clearColor;
			break;

		case BACKGROUND_MODE_GRADIENT:
			rayDir = optixGetWorldRayDirection();

			/* Dot rayDir with the up vector and use the result to interpolate between the bottom and top gradient colors */
			t = max(dot(normalize(rayDir), make_float3(0.0f, 0.0f, 1.0f)), 0.0f);
			result = lerp(optixLaunchParams.gradientBottom, optixLaunchParams.gradientTop, t);
			break;

		case BACKGROUND_MODE_TEXTURE:
			rayDir = optixGetWorldRayDirection();

			/* Convert the input ray direction to UV coordinates to access the background texture */
			u = 0.5f * (1.0f + atan2(rayDir.x, rayDir.y) * M_1_PIf) + optixLaunchParams.backgroundRotation;
			v = atan2(length(make_float2(rayDir.x, rayDir.y)), rayDir.z) * M_1_PIf;

			tex = tex2D<float4>(optixLaunchParams.backgroundTexture, u, v);
			result = make_float3(tex.x, tex.y, tex.z);
			break;
		}

		prd.throughput *= result;
		prd.done = true;
	}
}