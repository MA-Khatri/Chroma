#pragma once

#include "math.cuh"

/* 
 * Permuted Congruential Generator (PCG): www.pcg-random.org
 */

namespace otx
{
    struct PCG 
    {
        inline __host__ __device__ PCG()
        {
            /* 
             * intentionally empty so we can use it in device vars that
             * don't allow dynamic initialization (ie, PRD) 
             */
        }

        inline __host__ __device__ PCG(uint32_t seed)
        {
            Init(seed);
        }

        inline __host__ __device__ void Init(uint32_t seed)
        {
            state = seed;
            NextRandom();
        }

        inline __host__ __device__ void NextRandom()
        {
            state = state * 747796405 + 2891336453;
            state = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
            state = (state >> 22) ^ state;
        }

        /* Alias for RandomValue() */
        inline __host__ __device__ float operator() ()
        {
            return RandomValue();
        }

        /* Return a random value in [0, 1) */
        inline __host__ __device__ float RandomValue()
        {
            NextRandom();
            return state / 4294967295.0; /* 4294967295.0 = 2^32 - 1*/
        }

        /* Return a random point *on the circumference* of a unit circle */
        inline __host__ __device__ float2 RandomOnUnitCircle()
        {
            return SampleOnUnitCircle(RandomValue());
        }

        /* Return a random point *in* a unit circle (disk) */
        inline __host__ __device__ float2 RandomInUnitDisk()
        {
            return SampleInUnitDiskConcentric(make_float2(RandomValue(), RandomValue()));
        }

        /* Return a random point on the surface of a unit sphere */
        inline __host__ __device__ float3 RandomOnUnitSphere()
        {
            return SampleOnUnitSphere(make_float2(RandomValue(), RandomValue()));
        }

        /* Return a random point on the surface of a unit hemisphere */
        inline __host__ __device__ float3 RandomOnUnitHemisphere()
        {
            return SampleOnUnitHemisphere(make_float2(RandomValue(), RandomValue()));
        }

        /* Return a cosine weighted random point on the surface of a unit hemisphere */
        inline __host__ __device__ float3 RandomOnUnitCosineHemisphere()
        {
            return SampleOnUnitCosineHemisphere(make_float2(RandomValue(), RandomValue()));
        }

        uint32_t state;
    };

}