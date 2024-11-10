#pragma once

enum ControlMode
{
	CONTROL_MODE_FREE_FLY,
	CONTROL_MODE_ORBIT,
};

enum ProjectionMode
{
	PROJECTION_MODE_PERSPECTIVE,
	PROJECTION_MODE_ORTHOGRAPHIC,
	PROJECTION_MODE_THIN_LENS,
};

enum MaterialType
{
	MATERIAL_TYPE_LAMBERTIAN = 0,
	MATERIAL_TYPE_CONDUCTOR,
	MATERIAL_TYPE_DIELECTRIC,
	MATERIAL_TYPE_PRINCIPLED,
	MATERIAL_TYPE_DIFFUSE_LIGHT,
	MATERIAL_TYPE_COUNT
};

enum RayType
{
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_SHADOW,
	RAY_TYPE_COUNT
};

enum SamplerType
{
	SAMPLER_TYPE_INDEPENDENT = 0,
	SAMPLER_TYPE_STRATIFIED,
	SAMPLER_TYPE_MULTIJITTER
};

enum IntegratorType
{
	INTEGRATOR_TYPE_PATH = 0,
	// TODO, more...
};

enum LightType
{
	LIGHT_TYPE_AREA = 0, /* I.e., mesh lights, maybe later quad lights/sphere lights? */
	LIGHT_TYPE_DELTA, /* Point and spot lights */
	//LIGHT_TYPE_DIRECTIONAL, /* Directional, infinite area lights (TODO) */
	//LIGHT_TYPE_PORTAL, /* E.g., for sampling backgrounds through windows, etc. (TODO) */
	LIGHT_TYPE_COUNT
};

enum BackgroundMode
{
	BACKGROUND_MODE_SOLID_COLOR,
	BACKGROUND_MODE_GRADIENT,
	BACKGROUND_MODE_ENVIRONMENT_MAP
};

enum BlendMode
{
	BLEND_MODE_LINEAR,
	// TODO, more...
};