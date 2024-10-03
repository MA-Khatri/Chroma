#include "optix_renderer.h"

#include <fstream>

/* Note: this can only be included in one source file */
#include <optix_function_table_definition.h>

#ifdef _DEBUG
#define Debug(x) std::cout << x << std::endl;
bool debug_mode = true;
#else
#define Debug(x)
bool debug_mode = false;
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
		Debug("[Optix] Initializing Optix...");
		InitOptix();

		Debug("[Optix] Creating context...");
		CreateContext();

		Debug("[Optix] Setting up module...");
		CreateModule();

		Debug("[Optix] Creating raygen programs...");
		CreateRaygenPrograms();

		Debug("[Optix] Creating miss programs...");
		CreateMissPrograms();

		Debug("[Optix] Creating hitgroup programs...");
		CreateHitgroupPrograms();

		Debug("[Optix] Building acceleration structures...");
		m_LaunchParams.traversable = BuildAccel(CreatePlane());
		
		Debug("[Optix] Setting up Optix pipeline...");
		CreatePipeline();

		Debug("[Optix] Building SBT...");
		BuildSBT();

		Debug("\033[1;32m[Optix] Optix fully set up!\033[0m");

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


	/* The callback function for the Optix context (set in CreateContext) */
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
#ifdef _DEBUG
		OPTIX_CHECK(optixDeviceContextSetLogCallback(m_OptixContext, context_log_cb, nullptr, 4));
#else
		OPTIX_CHECK(optixDeviceContextSetLogCallback(m_OptixContext, context_log_cb, nullptr, 2));
#endif
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


	OptixTraversableHandle Optix::BuildAccel(const Mesh& mesh)
	{
		/* Upload mesh data to device */
		m_VertexBuffer.alloc_and_upload(mesh.vertices);
		m_IndexBuffer.alloc_and_upload(mesh.indices);

		OptixTraversableHandle asHandle{ 0 };

		/* ======================= */
		/* === Triangle inputs === */
		/* ======================= */
		OptixBuildInput triangleInput = {};
		triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		
		/* Create local variables to store pointers to the device pointers */
		CUdeviceptr d_vertices = m_VertexBuffer.d_pointer();
		CUdeviceptr d_indices = m_IndexBuffer.d_pointer();

		/* Set up format for reading vertex and index data */
		triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
		triangleInput.triangleArray.numVertices = (int)mesh.vertices.size();
		triangleInput.triangleArray.vertexBuffers = &d_vertices;

		triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
		triangleInput.triangleArray.numIndexTriplets = (int)mesh.indices.size();
		triangleInput.triangleArray.indexBuffer = d_indices;

		/* For now, we only have one SBT entry and no per-primitive materials */
		uint32_t triangleInputFlags[1] = { 0 };
		triangleInput.triangleArray.flags = triangleInputFlags;
		triangleInput.triangleArray.numSbtRecords = 1;
		triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

		/* ================== */
		/* === BLAS Setup === */
		/* ================== */
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_OptixContext, &accelOptions, &triangleInput, 1, &blasBufferSizes));

		/* ========================== */
		/* === Prepare compaction === */
		/* ========================== */
		CUDABuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.d_pointer();

		/* ===================== */
		/* === Execute build === */
		/* ===================== */
		CUDABuffer tempBuffer;
		tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

		CUDABuffer outputBuffer;
		outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

		OPTIX_CHECK(optixAccelBuild(m_OptixContext, 0, &accelOptions, &triangleInput, 1, tempBuffer.d_pointer(), tempBuffer.sizeInBytes, outputBuffer.d_pointer(), outputBuffer.sizeInBytes, &asHandle, &emitDesc, 1));
		CUDA_SYNC_CHECK();

		/* ========================== */
		/* === Perform compaction === */
		/* ========================== */
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);

		m_ASBuffer.alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(m_OptixContext, 0, asHandle, m_ASBuffer.d_pointer(), m_ASBuffer.sizeInBytes, &asHandle));
		CUDA_SYNC_CHECK();

		/* ================ */
		/* === Clean up === */
		/* ================ */
		outputBuffer.free(); /* Free the temporary, uncompacted buffer */
		tempBuffer.free();
		compactedSizeBuffer.free();

		return asHandle;
	}


	void Optix::Resize(const ImVec2& newSize)
	{
		/* If window is minimized */
		if (newSize.x == 0 || newSize.y == 0) return;

		/* Resize our CUDA framebuffer */
		m_ColorBuffer.resize(static_cast<size_t>(newSize.x * newSize.y * sizeof(uint32_t)));

		/* Update our launch parameters */
		m_LaunchParams.frame.size.x = static_cast<int>(newSize.x);
		m_LaunchParams.frame.size.y = static_cast<int>(newSize.y);
		m_LaunchParams.frame.colorBuffer = (uint32_t*)m_ColorBuffer.d_ptr;

		/* Reset the camera because our aspect ratio may have changed */
		SetCamera(m_LastSetCamera);
	}


	void Optix::SetCamera(const Camera& camera)
	{
		m_LastSetCamera = camera;
		m_LaunchParams.camera.position = glm::vec3(camera.position.x, camera.position.y, -camera.position.z);
		m_LaunchParams.camera.direction = glm::normalize(glm::vec3(camera.orientation.x, camera.orientation.y, -camera.orientation.z));
		const float cos_vfov = glm::cos(glm::radians(camera.vfov));
		const float aspect = m_LaunchParams.frame.size.x / float(m_LaunchParams.frame.size.y);
		m_LaunchParams.camera.horizontal = cos_vfov * aspect * glm::normalize(glm::cross(m_LaunchParams.camera.direction, camera.up));
		m_LaunchParams.camera.vertical = cos_vfov * glm::normalize(glm::cross(m_LaunchParams.camera.horizontal, m_LaunchParams.camera.direction));
	}


	void Optix::Render()
	{
		/* Sanity check: make sure we launch only after first resize is already done */
		if (m_LaunchParams.frame.size.x == 0 || m_LaunchParams.frame.size.y == 0) return;

		m_LaunchParamsBuffer.upload(&m_LaunchParams, 1);
		m_LaunchParams.frameID++;

		OPTIX_CHECK(optixLaunch(m_Pipeline, m_Stream, m_LaunchParamsBuffer.d_pointer(), m_LaunchParamsBuffer.sizeInBytes, &m_SBT, m_LaunchParams.frame.size.x, m_LaunchParams.frame.size.y, 1));

		/*
		 * Make sure frame is rendered before we display. BUT -- Vulkan does not know when this is finished!
		 * For higher performance, we should use streams and do double-buffering
		 */
		CUDA_SYNC_CHECK();
	}


	void Optix::DownloadPixels(uint32_t h_pixels[])
	{
		m_ColorBuffer.download(h_pixels, m_LaunchParams.frame.size.x * m_LaunchParams.frame.size.y);
	}
}