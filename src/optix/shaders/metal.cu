#include "utils.cuh"

namespace otx
{
	extern "C" __global__ void __closesthit__radiance__metal()
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
		float3 Ng = cross(v1 - v0, v2 - v0);
		float3 Ns = (sbtData.normal) ? InterpolateNormals(uv, sbtData.normal[index.x], sbtData.normal[index.y], sbtData.normal[index.z]) : Ng;

		/* Compute world-space normal and normalize */
		Ns = normalize(optixTransformNormalFromObjectToWorldSpace(Ns));

		/* Face forward normal */
		if (dot(rayDir, Ns) > 0.0f) Ns = -Ns;

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

		/* If we hit a light, stop tracing */
		if (diffuseColor.x > 1.0f || diffuseColor.y > 1.0f || diffuseColor.z > 1.0f) prd.done = true;

		/* === Set ray data for next trace call === */
		/* Determine reflected ray origin and direction */
		float3 reflectDir = reflect(rayDir, Ns); /* Reflected ray direction */
		float3 reflectOrigin = HitPosition() + 1e-3f * Ns;
		prd.origin = reflectOrigin;
		prd.direction = reflectDir;
	}


	extern "C" __global__ void __anyhit__radiance__metal()
	{
		// TODO?
	}
}