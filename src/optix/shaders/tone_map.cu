#include "tone_map.cuh"

#include <cuda_runtime.h>
#include <vector_math.h>

/* 
 * Based on: https://github.com/ingowald/optix7course/blob/29e279745161003cf905a8a4f1fb603268d48efa/example12_denoiseSeparateChannels/toneMap.cu 
 */

namespace otx
{
    inline __device__ float4 sqrt(float4 f)
    {
        return make_float4(sqrtf(f.x), sqrtf(f.y), sqrtf(f.z), sqrtf(f.w));
    }

    inline __device__ float  clampf(float f)
    { 
        return fminf(1.f, fmaxf(0.0f, f)); 
    }
    
    inline __device__ float4 clamp(float4 f)
    {
        return make_float4(clampf(f.x),
            clampf(f.y),
            clampf(f.z),
            clampf(f.w));
    }

    /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
    __global__ void computeFinalPixelColorsKernel(uint32_t* finalColorBuffer, float4* denoisedBuffer, int2 size, bool gammaCorrect)
    {
        int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
        int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
        if (pixelX >= size.x) return;
        if (pixelY >= size.y) return;

        int pixelID = pixelX + size.x * pixelY;

        float4 f4 = denoisedBuffer[pixelID];
        if (gammaCorrect) f4 = clamp(sqrt(f4));
        else f4 = clamp(f4);
        uint32_t rgba = 0;
        rgba |= (uint32_t)(f4.x * 255.9f) << 0;
        rgba |= (uint32_t)(f4.y * 255.9f) << 8;
        rgba |= (uint32_t)(f4.z * 255.9f) << 16;
        rgba |= (uint32_t)255 << 24;
        finalColorBuffer[pixelID] = rgba;
    }

    void ComputeFinalPixelColors(int2 fbSize, uint32_t* finalColorBuffer, float4* denoisedBuffer, bool gammaCorrect)
    {
        int2 blockSize = make_int2(32);
        int2 numBlocks = make_int2((fbSize.x + blockSize.x - 1) / blockSize.x, (fbSize.y + blockSize.y - 1) / blockSize.y);

        computeFinalPixelColorsKernel<<<dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y)>>>(finalColorBuffer, denoisedBuffer, fbSize, gammaCorrect);
    }
}