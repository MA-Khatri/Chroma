#include "utils.cuh"

namespace otx
{
	extern "C" __global__ void __closesthit__shadow()
	{
		/* Not going to be used... */
	}

	extern "C" __global__ void __anyhit__shadow()
	{
		/* Not going to be used... */
	}

	extern "C" __global__ void __miss__shadow()
	{
		/* Nothing was hit so the light is visible */
		float3& prd = *(float3*)getPRD<float3>();
		prd = make_float3(1.0f);
	}
}