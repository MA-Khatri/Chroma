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
		PRD_Shadow& prd = *(PRD_Shadow*)getPRD<PRD_Shadow>();
		prd.radiance = make_float3(1.0f); /* Not used at the moment! */
		prd.reachedLight = true;
	}
}