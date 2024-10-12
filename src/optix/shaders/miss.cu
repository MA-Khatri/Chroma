#include "utils.cuh"

namespace otx
{
	extern "C" __global__ void __miss__radiance()
	{
		PRD_radiance& prd = *getPRD<PRD_radiance>();

		///* Get brighter as you approach +z */
		//float3 rayDir = optixGetWorldRayDirection();
		//const float t = max(dot(normalize(rayDir), make_float3(0.0f, 0.0f, 1.0f)), 0.0f);
		//const float3 result = lerp(make_float3(0.3f), make_float3(1.0f), t);

		const float3 result = make_float3(0.0f);

		prd.radiance *= result;
		prd.done = true;
	}
}