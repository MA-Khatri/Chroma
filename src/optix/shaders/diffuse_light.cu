#include "utils.cuh"

namespace otx
{
	extern "C" __global__ void __closesthit__radiance()
	{
		const MeshSBTData& sbtData = *(const MeshSBTData*)optixGetSbtDataPointer();
		PRD_radiance& prd = *getPRD<PRD_radiance>();

		const int primID = optixGetPrimitiveIndex();
		const int3 index = sbtData.index[primID];
		float2 uv = optixGetTriangleBarycentrics();

		/* Use the stored color in the sbtData as the brightness of the light */
		float3 lightColor = *sbtData.color;

		/* If this light has a diffuse texture, scale the lightColor by that texture (creating a textured light) */
		float2 tc = TexCoord(uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			lightColor *= make_float3(tex.x, tex.y, tex.z);
		}
		prd.radiance *= lightColor;

		/* Terminate ray */
		prd.done = true;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}
}