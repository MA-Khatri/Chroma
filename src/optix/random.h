#pragma once

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
            float angle = RandomValue() * 2.0f * M_PIf;
            return make_float2(cos(angle), sin(angle));
        }

        /* Return a random point *in* a unit circle (disk) by first generating polar coordinates */
        inline __host__ __device__ float2 RandomInUnitDisk()
        {
            float r = sqrtf(RandomValue());
            float t = 2 * M_PIf * RandomValue();
            return make_float2(r * cos(t), r * sin(t));
        }

        uint32_t state;
    };


    ///* 
    // * Simple 24-bit linear congruence generator
    // * From https://github.com/ingowald/optix7course/blob/master/common/gdt/gdt/random/random.h 
    // */
    //template<unsigned int N = 16>
    //struct LCG 
    //{
    //    inline __host__ __device__ LCG()
    //    { /* intentionally empty so we can use it in device vars that
    //         don't allow dynamic initialization (ie, PRD) */
    //    }
    //    inline __host__ __device__ LCG(unsigned int val0, unsigned int val1)
    //    {
    //        init(val0, val1);
    //    }

    //    inline __host__ __device__ void init(unsigned int val0, unsigned int val1)
    //    {
    //        unsigned int v0 = val0;
    //        unsigned int v1 = val1;
    //        unsigned int s0 = 0;

    //        for (unsigned int n = 0; n < N; n++) {
    //            s0 += 0x9e3779b9;
    //            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    //            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    //        }
    //        state = v0;
    //    }

    //    /* Generate random unsigned int in[0, 2 ^ 24) */
    //    inline __host__ __device__ float operator() ()
    //    {
    //        const uint32_t LCG_A = 1664525u;
    //        const uint32_t LCG_C = 1013904223u;
    //        state = (LCG_A * state + LCG_C);
    //        return (state & 0x00FFFFFF) / (float)0x01000000;
    //    }

    //    uint32_t state;
    //};

}