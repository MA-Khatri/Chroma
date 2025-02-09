#pragma once

enum class ControlMode
{
	FREE_FLY,
	ORBIT,
};

enum class ProjectionMode
{
	PERSPECTIVE,
	ORTHOGRAPHIC,
	THIN_LENS,
};

enum class MaterialType
{
	LAMBERTIAN = 0,
	CONDUCTOR,
	DIELECTRIC,
	PRINCIPLED,
	DIFFUSE_LIGHT,
	COUNT
};

enum class RayType
{
	RADIANCE = 0,
	SHADOW,
	COUNT
};

enum class SamplerType
{
	INDEPENDENT = 0,
	STRATIFIED,
	MULTIJITTER
};

enum class IntegratorType
{
	PATH = 0,
	// TODO, more...
};

enum class LightType
{
	AREA = 0, /* I.e., mesh lights, maybe later quad lights/sphere lights? */
	DELTA, /* Point and spot lights */
	// DIRECTIONAL, /* Directional, infinite area lights (TODO) */
	// PORTAL, /* E.g., for sampling backgrounds through windows, etc. (TODO) */
	COUNT
};

enum class BackgroundMode
{
	SOLID_COLOR,
	GRADIENT,
	ENVIRONMENT_MAP
};

enum class BlendMode
{
	LINEAR,
	// TODO, more...
};