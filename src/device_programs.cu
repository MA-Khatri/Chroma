#include <optix_device.h>

#include "launch_params.h"

namespace otx
{
	/* Launch parameters in constant memory, filled in by Optix upon optixLaunch */
	extern "C" __constant__ LaunchParams optixLaunchParams;

	/*
	 * Closest hit and any hit programs for radiance-type rays.
	 * Eventually, we will need a pair of these for each ray type 
	 * and geometry type that we want to render.
	 */
	extern "C" __global__ void __closesthit__radiance()
	{
		// TODO
	}

	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO
	}


	/* 
	 * Miss program that gets called for any ray that did not have a valid intersection.
	 */
	extern "C" __global__ void __miss__radiance()
	{
		// TODO
	}


	/*
	 * The primary ray gen program where rendering happens.
	 */
	extern "C" __global__ void __raygen__renderFrame()
	{
		if (optixLaunchParams.frameID == 0 &&
			optixGetLaunchIndex().x == 0 &&
			optixGetLaunchIndex().y == 0) {
			// we could of course also have used optixGetLaunchDims to query
			// the launch size, but accessing the optixLaunchParams here
			// makes sure they're not getting optimized away (because
			// otherwise they'd not get used)
			printf("############################################\n");
			printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
				optixLaunchParams.fbSize.x,
				optixLaunchParams.fbSize.y);
			printf("############################################\n");
		}

		// ------------------------------------------------------------------
		// for this example, produce a simple test pattern:
		// ------------------------------------------------------------------

		// compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		const int r = (ix % 256);
		const int g = (iy % 256);
		const int b = ((ix + iy) % 256);

		// convert to 32-bit rgba value (we explicitly set alpha to 0xff
		// to make stb_image_write happy ...
		const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

		// and write to frame buffer ...
		const uint32_t fbIndex = ix + iy * optixLaunchParams.fbSize.x;
		optixLaunchParams.colorBuffer[fbIndex] = rgba;
	}
}