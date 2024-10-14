#pragma once

namespace otx
{
	enum MaterialType
	{
		MATERIAL_TYPE_LAMBERTIAN = 0,
		MATERIAL_TYPE_CONDUCTOR,
		MATERIAL_TYPE_DIELECTRIC,
		MATERIAL_TYPE_DIFFUSE_LIGHT,
		MATERIAL_TYPE_COUNT
	};

	enum RayType
	{
		RAY_TYPE_RADIANCE = 0,
		RAY_TYPE_SHADOW,
		RAY_TYPE_COUNT
	};
}