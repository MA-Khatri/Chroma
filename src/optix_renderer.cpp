#include "optix_renderer.h"

#include <fstream>

/* Note: this can only be included in one source file */
#include <optix_function_table_definition.h>

bool debug_mode = false;
#ifdef _DEBUG
#define Debug(x) std::cout << x << std::endl;
debug_mode = true;
#else
#define Debug(x)
#endif


namespace otx
{
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
		InitOptix();
		CreateContext();
		CreateModule();
		CreateRaygenPrograms();
		CreateMissPrograms();
		CreateHitgroupPrograms();
		CreatePipeline();
		BuildSBT();

		m_LaunchParamsBuffer.alloc(sizeof(m_LaunchParams));
	}


	void Optix::InitOptix()
	{
		/* Check for Optix capable devices */
		cudaFree(0);

		int numDevices;
		cudaGetDeviceCount(&numDevices);
		if (numDevices == 0)
		{
			std::cerr << "InitOptix(): no CUDA capable devices found!" << std::endl;
			exit(-1);
		}

		/* Initialize Optix */
		OPTIX_CHECK(optixInit());
	}


	/* The callback function for the Optix context (only used in CreateContext) */
	static void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
	{
		fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
	}

	void Optix::CreateContext()
	{
		const int deviceID = 0;
		CUDA_CHECK(SetDevice(deviceID));
		CUDA_CHECK(StreamCreate(&m_Stream));

		cudaGetDeviceProperties(&m_DeviceProps, deviceID);
		Debug("Optix Running on device " << m_DeviceProps.name);

		CUresult cuRes = cuCtxGetCurrent(&m_CudaContext);
		if (cuRes != CUDA_SUCCESS)
		{
			fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
		}

		OPTIX_CHECK(optixDeviceContextCreate(m_CudaContext, 0, &m_OptixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(m_OptixContext, context_log_cb, nullptr, 4));
	}


	void Optix::CreateModule()
	{
		m_ModuleCompileOptions.maxRegisterCount = 50;
#ifdef _DEBUG
		m_ModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		m_ModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
		m_ModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		m_ModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

		m_PipelineCompileOptions = {};
		m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		m_PipelineCompileOptions.usesMotionBlur = false;
		m_PipelineCompileOptions.numPayloadValues = 2;
		m_PipelineCompileOptions.numAttributeValues = 2;
		m_PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

		m_PipelineLinkOptions.maxTraceDepth = 2;

		std::ifstream input("src/device_programs.optixir", std::ios::binary);
		std::vector<char> ptxCode(std::istreambuf_iterator<char>(input), {});
		if (ptxCode.empty())
		{
			std::cerr << "Optix::CreateModule(): Failed to load optixir code!" << std::endl;
			exit(-1);
		}

		char log[2048];
		size_t sizeof_log = sizeof(log);
#if OPTIX_VERSION >= 70700
		OPTIX_CHECK(optixModuleCreate(m_OptixContext, &m_ModuleCompileOptions, &m_PipelineCompileOptions, ptxCode.data(), ptxCode.size(), log, &sizeof_log, &m_Module));
#else
		OPTIX_CHECK(optixModuleCreateFromPTX(m_OptixContext, &m_ModuleCompileOptions, &m_PipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &m_Module));
#endif
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
	}


	void Optix::CreateRaygenPrograms()
	{
		m_RaygenPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pgDesc.raygen.module = m_Module;
		pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(m_OptixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &m_RaygenPGs[0]));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
	}


	void Optix::CreateMissPrograms()
	{
		m_MissPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		pgDesc.miss.module = m_Module;
		pgDesc.miss.entryFunctionName = "__miss__radiance";

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(m_OptixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &m_MissPGs[0]));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
	}


	void Optix::CreateHitgroupPrograms()
	{
		m_HitgroupPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		pgDesc.hitgroup.moduleCH = m_Module;
		pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		pgDesc.hitgroup.moduleAH = m_Module;
		pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(m_OptixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &m_HitgroupPGs[0]));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
	}


	void Optix::CreatePipeline()
	{
		std::vector<OptixProgramGroup> programGroups;
		for (auto pg : m_RaygenPGs)
		{
			programGroups.push_back(pg);
		}
		for (auto pg : m_MissPGs)
		{
			programGroups.push_back(pg);
		}
		for (auto pg : m_HitgroupPGs)
		{
			programGroups.push_back(pg);
		}

		char log[2048];
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK(optixPipelineCreate(m_OptixContext, &m_PipelineCompileOptions, &m_PipelineLinkOptions, programGroups.data(), (int)programGroups.size(), log, &sizeof_log, &m_Pipeline));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;

		OPTIX_CHECK(optixPipelineSetStackSize(
			m_Pipeline, /* [in] The pipeline to configure the stack size for */
			2 * 1024,   /* [in] The direct stack size requirement for direct callables invoked from IS or AH */
			2 * 1024,   /* [in] The direct stack size requirement for direct callables invoked from RG, MS, or CH */
			2 * 1024,   /* [in] The continuation stack size requirement */
			1		    /* [in] The maximum depth of a traversable graph passed to trace */
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
	}


	void Optix::BuildSBT()
	{
		/* Build raygen records */
		std::vector<RaygenRecord> raygenRecords;
		for (int i = 0; i < m_RaygenPGs.size(); i++)
		{
			RaygenRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(m_RaygenPGs[i], &rec));
			rec.data = nullptr; /* for now... */
			raygenRecords.push_back(rec);
		}
		m_RaygenRecordsBuffer.alloc_and_upload(raygenRecords);
		m_SBT.raygenRecord = m_RaygenRecordsBuffer.d_pointer();

		/* Build miss records */
		std::vector<MissRecord> missRecords;
		for (int i = 0; i < m_MissPGs.size(); i++)
		{
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(m_MissPGs[i], &rec));
			rec.data = nullptr; /* for now... */
			missRecords.push_back(rec);
		}
		m_MissRecordsBuffer.alloc_and_upload(missRecords);
		m_SBT.missRecordBase = m_MissRecordsBuffer.d_pointer();
		m_SBT.missRecordStrideInBytes = sizeof(MissRecord);
		m_SBT.missRecordCount = (int)missRecords.size();

		/* Build hitgroup records */
		int numObjects = 1;
		std::vector<HitgroupRecord> hitgroupRecords;
		for (int i = 0; i < numObjects; i++)
		{
			int objectType = 0;
			HitgroupRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(m_HitgroupPGs[objectType], &rec));
			rec.objectID = i;
			hitgroupRecords.push_back(rec);
		}
		m_HitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
		m_SBT.hitgroupRecordBase = m_HitgroupRecordsBuffer.d_pointer();
		m_SBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		m_SBT.hitgroupRecordCount = (int)hitgroupRecords.size();
	}


	void Optix::Render()
	{
		/* Sanity check: make sure we launch only after first resize is already done */
		if (m_LaunchParams.fbWidth == 0 || m_LaunchParams.fbHeight == 0) return;

		m_LaunchParamsBuffer.upload(&m_LaunchParams, 1);
		m_LaunchParams.frameID++;

		OPTIX_CHECK(optixLaunch(m_Pipeline, m_Stream, m_LaunchParamsBuffer.d_pointer(), m_LaunchParamsBuffer.sizeInBytes, &m_SBT, m_LaunchParams.fbWidth, m_LaunchParams.fbHeight, 1));

		/*
		 * Make sure frame is rendered before we display.
		 * For higher performance, we should use streams and do double-buffering
		 */
		CUDA_SYNC_CHECK();
	}


	void Optix::Resize(const ImVec2& newSize)
	{
		/* If window is minimized */
		if (newSize.x == 0 || newSize.y == 0) return;

		/* Resize our CUDA framebuffer */
		m_ColorBuffer.resize(static_cast<size_t>(newSize.x * newSize.y * sizeof(uint32_t)));

		/* Update our launch parameters */
		m_LaunchParams.fbWidth = static_cast<int>(newSize.x);
		m_LaunchParams.fbHeight = static_cast<int>(newSize.y);
		m_LaunchParams.colorBuffer = (uint32_t*)m_ColorBuffer.d_ptr;
	}


	void Optix::DownloadPixels(uint32_t h_pixels[])
	{
		m_ColorBuffer.download(h_pixels, m_LaunchParams.fbWidth * m_LaunchParams.fbHeight);
	}
}