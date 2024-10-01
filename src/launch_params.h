#pragma once

#include "glm/glm.hpp"

namespace otx
{
	struct LaunchParams
	{
		int frameID{ 0 };
		uint32_t* colorBuffer;
		glm::ivec2 fbSize;
	};
}
