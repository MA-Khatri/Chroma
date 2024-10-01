#include "optix_renderer.h"

/* Note: this can only be included in one source file */
#include <optix_function_table_definition.h>


/* Debugging macros setup */
#define PRINT_DEBUG 1

#ifdef _DEBUG && PRINT_DEBUG
#define Debug( x ) std::cout << "[Optix] " << x << std::endl;
#define DebugR( x ) std::cout << "\033[1;31m[Optix] " << x << "\033[0m" << std::endl; /* Red */
#define DebugG( x ) std::cout << "\033[1;32m[Optix] " << x << "\033[0m" << std::endl; /* Green */
#define DebugY( x ) std::cout << "\033[1;33m[Optix] " << x << "\033[0m" << std::endl; /* Yellow */
#define DebugB( x ) std::cout << "\033[1;34m[Optix] " << x << "\033[0m" << std::endl; /* Blue */
#define DEBUG( x ) std::cout << "\033[1;1m[Optix] " << x << "\033[0m" << std::endl; /* BOLD */
#else
#define Debug( x )
#define DebugR( x )
#define DebugG( x )
#define DebugY( x )
#define DebugB( x )
#define DEBUG( x )
#endif
#define ERROR( x ) std::cerr << "\033[1;31m[Optix] " << x << "\033[0m" << std::endl; /* prints in red */


namespace otx
{
	extern "C" char embedded_ptx_code[];

	/* SBT record for a raygen program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		void* data;
	};

	/* SBT record for a miss program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		void* data;
	};

	/* SBT record for a hitgroup program */
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		int objectID;
	};


	Optix::Optix()
	{
		Debug("Initializing Optix...");
		InitOptix();

		Debug("Creating Optix context...");
		CreateContext();

		Debug("Setting up module...");
		CreateModule();

		Debug("Creating raygen programs...");
		CreateRaygenPrograms();

		Debug("Creating miss programs...");
		CreateMissPrograms();

		Debug("Creating hitgroup programs...");
		CreateHitgroupPrograms();

		Debug("Setting up Optix pipline...");
		CreatePipeline();

		Debug("Building shader binding table...");
		BuildSBT();

		m_LaunchParamsBuffer.alloc(sizeof(m_LaunchParams));
		DebugG("Optix setup complete!");
	}


	void Optix::InitOptix()
	{
		/* Check for Optix capable devices */
		cudaFree(0);

		int numDevices;
		cudaGetDeviceCount(&numDevices);
		if (numDevices == 0)
		{
			ERROR("InitOptix(): no CUDA capable devices found!");
			exit(-1);
		}

		/* Initialize Optix */
		OPTIX_CHECK(optixInit());
	}


	static void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
	{
		fprintf(stderr, "\033[1;31m[%2d][%12s]: %s\n\033[0m", (int)level, tag, message);
	}


	void Optix::CreateContext()
	{
		const int deviceID = 0;
		CUDA_CHECK(SetDevice(deviceID));
		CUDA_CHECK(StreamCreate(&m_Stream));

		cudaGetDeviceProperties(&m_DeviceProps, deviceID);
		DebugY("Running on device " << m_DeviceProps.name);

		CUresult cuRes = cuCtxGetCurrent(&m_CudaContext);
		if (cuRes != CUDA_SUCCESS)
		{
			ERROR("Error querying current context: error code " << std::to_string(cuRes));
		}
	}


	void Optix::CreateModule()
	{

	}


	void Optix::CreateRaygenPrograms()
	{

	}


	void Optix::CreateMissPrograms()
	{

	}


	void Optix::CreateHitgroupPrograms()
	{

	}


	void Optix::CreatePipeline()
	{

	}


	void Optix::BuildSBT()
	{

	}
}