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
		/* Nothing was hit so the light is visible -- return the radiance of the light */
		PRD_Shadow& prd = *(PRD_Shadow*)getPRD<PRD_Shadow>();
		prd.radiance = make_float3(20.0f);
		prd.reachedLight = true;
	}
}