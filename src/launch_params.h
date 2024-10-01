#pragma once

namespace otx
{
	
	struct LaunchParams
	{
		int frameID{ 0 };
		uint32_t* colorBuffer;
		int fbWidth;
		int fbHeight;
	};

} /* namspace otx */
