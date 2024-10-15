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
		float3 rayDir = optixGetWorldRayDirection();

		/* === Compute normal === */
		/* Use shading normal if available, else use geometry normal */
		const float3& v0 = sbtData.position[index.x];
		const float3& v1 = sbtData.position[index.y];
		const float3& v2 = sbtData.position[index.z];
		float3 N = (sbtData.normal) ? InterpolateNormals(uv, sbtData.normal[index.x], sbtData.normal[index.y], sbtData.normal[index.z]) : cross(v1 - v0, v2 - v0);

		/* Compute world-space normal and normalize */
		N = normalize(optixTransformNormalFromObjectToWorldSpace(N));

		/* Face forward normal */
		if (dot(rayDir, N) > 0.0f) N = -N;

		/* Default diffuse color if no diffuse texture */
		float3 diffuseColor = *sbtData.color;

		/* === Sample diffuse texture === */
		float2 tc = TexCoord(uv, sbtData.texCoord[index.x], sbtData.texCoord[index.y], sbtData.texCoord[index.z]);
		if (sbtData.hasDiffuseTexture)
		{
			float4 tex = tex2D<float4>(sbtData.diffuseTexture, tc.x, tc.y);
			diffuseColor = make_float3(tex.x, tex.y, tex.z);
		}
		prd.radiance *= diffuseColor;

		/* === Set ray data for next trace call === */
		/* Determine reflected ray origin and direction */
		OrthonormalBasis basis = OrthonormalBasis(N);
		float3 reflectDir = basis.Local(prd.random.RandomOnUnitCosineHemisphere());
		float3 reflectOrigin = FrontHitPosition(N);
		prd.origin = reflectOrigin;
		prd.direction = reflectDir;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}
}