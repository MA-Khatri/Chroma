#include "utils.cuh"

namespace otx
{
	__forceinline__ __device__ float3 Sample(PRD_Radiance& prd)
	{
		// TODO
	}

	__forceinline__ __device__ float3 Eval(PRD_Radiance& prd, float3 indir, float3 outdir)
	{
		// TODO
	}


	__forceinline__ __device__ float PDF(PRD_Radiance& prd, float3 w)
	{
		// TODO
	}


	extern "C" __global__ void __closesthit__radiance()
	{
		// TODO
	}




	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}


	extern "C" __device__ float3 __direct_callable__sample(PRD_Radiance & prd)
	{
		return Sample(prd);
	}


	extern "C" __device__ float3 __direct_callable__eval(PRD_Radiance & prd, float3 indir, float3 outdir)
	{
		return Eval(prd, indir, outdir);
	}


	extern "C" __device__ float __direct_callable__pdf(PRD_Radiance & prd, float3 w)
	{
		return PDF(prd, w);
	}
}