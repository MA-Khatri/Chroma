#include <cuda_runtime.h>
#include <cuda.h>
#include <vector_math.h>

namespace otx
{
	void ComputeFinalPixelColors(int2 fbSize, uint32_t* finalColorBuffer, float4* denoisedBuffer, CUstream stream, bool gammaCorrect);
}