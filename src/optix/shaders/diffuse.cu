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
		OrthonormalBasis basis = OrthonormalBasis(Ns);
		float3 reflectDir = basis.Local(prd.random.RandomOnUnitCosineHemisphere());
		float3 reflectOrigin = HitPosition() + 1e-3f * Ns;
		prd.origin = reflectOrigin;
		prd.direction = reflectDir;

		///* === Compute shadow === */
		//const float3 surfPosn = HitPosition();
		//const float3 lightPosn = make_float3(100.0f, 100.0f, 100.0f); /* Hard coded light position (for now) */
		//const float3 lightDir = lightPosn - surfPosn;

		///* Trace shadow ray*/
		//float3 lightVisibility = make_float3(0.5f);
		//uint32_t u0, u1;
		//packPointer(&lightVisibility, u0, u1);
		//optixTrace(
		//	optixLaunchParams.traversable,
		//	surfPosn + 1e-3f * Ns,
		//	lightDir,
		//	1e-3f, /* tmin */
		//	1.0f-1e-3f, /* tmax -- in terms of lightDir length */
		//	0.0f, /* ray time */
		//	OptixVisibilityMask(255),
		//	OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		//	SHADOW_RAY_TYPE, /* SBT offset */
		//	RAY_TYPE_COUNT, /* SBT stride */
		//	SHADOW_RAY_TYPE, /* missSBT index */
		//	u0, u1 /* packed pointer to our PRD */
		//);


		/////* Calculate shading */
		////const float ambient = 0.2f;
		////const float diffuse = 0.5f;
		////const float specular = 0.1f;
		////const float exponent = 16.0f;

		////const float3 reflectDir = reflect(rayDir, Ns);
		////const float diffuseContrib = clamp(dot(-rayDir, Ns), 0.0f, 1.0f);
		////const float specularContrib = pow(max(dot(-rayDir, reflectDir), 0.0f), exponent);
		////const float lc = ambient + diffuse * diffuseContrib + specular * specularContrib;

		//prd.radiance *= lightVisibility;
	}


	extern "C" __global__ void __anyhit__radiance()
	{
		// TODO?
	}
}