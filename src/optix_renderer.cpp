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
		MeshSBTData data;
	};


	Optix::Optix(std::shared_ptr<Scene> scene)
		:m_Scene(scene)
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
		m_LaunchParams.traversable = BuildAccel();
		
		Debug("[Optix] Setting up Optix pipeline...");
		CreatePipeline();

		Debug("[Optix] Creating textures...");
		CreateTextures();

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
		OPTIX_CHECK(optixModuleCreate(
			m_OptixContext,
			&m_ModuleCompileOptions,
			&m_PipelineCompileOptions, 
			ptxCode.data(),
			ptxCode.size(),
			log,
			&sizeof_log,
			&m_Module
		));
#else
		OPTIX_CHECK(optixModuleCreateFromPTX(
			m_OptixContext, 
			&m_ModuleCompileOptions, 
			&m_PipelineCompileOptions,
			ptxCode.c_str(), 
			ptxCode.size(), 
			log, 
			&sizeof_log, 
			&m_Module
		));
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
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log, 
			&sizeof_log, 
			&m_RaygenPGs[0]
		));
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
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log, 
			&sizeof_log, 
			&m_MissPGs[0]
		));
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
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log, 
			&sizeof_log, 
			&m_HitgroupPGs[0]
		));
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
		OPTIX_CHECK(optixPipelineCreate(
			m_OptixContext, 
			&m_PipelineCompileOptions, 
			&m_PipelineLinkOptions, 
			programGroups.data(), 
			(int)programGroups.size(), 
			log, 
			&sizeof_log, 
			&m_Pipeline
		));
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


	OptixTraversableHandle Optix::BuildAccel()
	{
		auto& objects = m_Scene->m_Objects;
		int nObjects = objects.size();

		/* Extract meshes from scene objects */
		m_Meshes.reserve(nObjects);
		for (auto& object : objects)
		{
			if (object->m_RayTraceRender)
			{
				m_Meshes.emplace_back(object->m_Mesh);
			}
		}
		m_Meshes.shrink_to_fit();
		int nMeshes = m_Meshes.size();

		/* Upload mesh data to device */
		m_VertexBuffers.resize(nMeshes);
		m_IndexBuffers.resize(nMeshes);
		m_NormalBuffers.resize(nMeshes);
		m_TexCoordBuffers.resize(nMeshes);

		OptixTraversableHandle asHandle{ 0 };

		/* ======================= */
		/* === Triangle inputs === */
		/* ======================= */
		std::vector<OptixBuildInput> triangleInputs(nMeshes);
		std::vector<CUdeviceptr> d_vertices(nMeshes);
		std::vector<CUdeviceptr> d_indices(nMeshes);
		std::vector<uint32_t> triangleInputFlags(nMeshes);

		for (int meshID = 0; meshID < nMeshes; meshID++)
		{
			/* Upload the mesh to the device */
			Mesh& mesh = m_Meshes[meshID];
			m_VertexBuffers[meshID].alloc_and_upload(mesh.posns);
			m_IndexBuffers[meshID].alloc_and_upload(mesh.ivecIndices);
			m_NormalBuffers[meshID].alloc_and_upload(mesh.normals);
			m_TexCoordBuffers[meshID].alloc_and_upload(mesh.texCoords);

			triangleInputs[meshID] = {};
			triangleInputs[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			/* Create local variables to store pointers to the device pointers */
			d_vertices[meshID] = m_VertexBuffers[meshID].d_pointer();
			d_indices[meshID] = m_IndexBuffers[meshID].d_pointer();

			/* Set up format for reading vertex and index data */
			triangleInputs[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInputs[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
			triangleInputs[meshID].triangleArray.numVertices = (int)mesh.posns.size();
			triangleInputs[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

			triangleInputs[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInputs[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
			triangleInputs[meshID].triangleArray.numIndexTriplets = (int)mesh.ivecIndices.size();
			triangleInputs[meshID].triangleArray.indexBuffer = d_indices[meshID];

			triangleInputFlags[meshID] = 0;

			/* For now, we only have one SBT entry and no per-primitive materials */
			triangleInputs[meshID].triangleArray.flags = &triangleInputFlags[meshID];
			triangleInputs[meshID].triangleArray.numSbtRecords = 1;
			triangleInputs[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
			triangleInputs[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
			triangleInputs[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
		}

		/* ================== */
		/* === BLAS Setup === */
		/* ================== */
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			m_OptixContext,
			&accelOptions,
			triangleInputs.data(),
			nMeshes,
			&blasBufferSizes
		));

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

		OPTIX_CHECK(optixAccelBuild(
			m_OptixContext,
			0, /* stream */
			&accelOptions,
			triangleInputs.data(),
			nMeshes,
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,
			outputBuffer.d_pointer(),
			outputBuffer.sizeInBytes,
			&asHandle,
			&emitDesc,
			1
		));
		CUDA_SYNC_CHECK();

		/* ========================== */
		/* === Perform compaction === */
		/* ========================== */
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);

		m_ASBuffer.alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(
			m_OptixContext,
			0,
			asHandle,
			m_ASBuffer.d_pointer(),
			m_ASBuffer.sizeInBytes,
			&asHandle
		));
		CUDA_SYNC_CHECK();

		/* ================ */
		/* === Clean up === */
		/* ================ */
		outputBuffer.free(); /* Free the temporary, uncompacted buffer */
		tempBuffer.free();
		compactedSizeBuffer.free();

		return asHandle;
	}


	void Optix::CreateTextures()
	{
		int nTextures = 0;
		for (auto obj : m_Scene->m_Objects)
		{
			if (obj->m_DiffuseTexture.pixels) nTextures++;
			if (obj->m_SpecularTexture.pixels) nTextures++;
			if (obj->m_NormalTexture.pixels) nTextures++;
		}

		m_TextureArrays.resize(nTextures);
		m_TextureObjects.resize(nTextures);

		int textureID = 0;
		for (auto obj : m_Scene->m_Objects)
		{
			if (!obj->m_RayTraceRender) continue;

			/* Get all textures for this object */
			std::vector<Texture*> textures;
			textures.reserve(3);

			Texture* diffuse = &(obj->m_DiffuseTexture);
			if (diffuse->pixels) { diffuse->textureID = textureID; textureID++; textures.emplace_back(diffuse); }

			Texture* specular = &(obj->m_SpecularTexture);
			if (specular->pixels) { specular->textureID = textureID; textureID++; textures.emplace_back(specular); }

			Texture* normal = &(obj->m_NormalTexture);
			if (normal->pixels) { normal->textureID = textureID; textureID++; textures.emplace_back(normal); }

			/* Create CUDA resources for each texture */
			for (Texture* tex : textures)
			{
				int32_t width = tex->resolution.x;
				int32_t height = tex->resolution.y;
				int32_t numComponents = tex->resolution.z;
				int32_t pitch = width * numComponents * sizeof(uint8_t);
				cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

				cudaArray_t& pixelArray = m_TextureArrays[tex->textureID];
				CUDA_CHECK(MallocArray(&pixelArray, &channel_desc, width, height));
				CUDA_CHECK(Memcpy2DToArray(pixelArray, 0, 0, tex->pixels, pitch, pitch, height, cudaMemcpyHostToDevice));

				cudaResourceDesc res_desc = {};
				res_desc.resType = cudaResourceTypeArray;
				res_desc.res.array.array = pixelArray;

				cudaTextureDesc tex_desc = {};
				tex_desc.addressMode[0] = cudaAddressModeWrap;
				tex_desc.addressMode[1] = cudaAddressModeWrap;
				tex_desc.filterMode = cudaFilterModeLinear;
				tex_desc.readMode = cudaReadModeNormalizedFloat;
				tex_desc.normalizedCoords = 1;
				tex_desc.maxAnisotropy = 1;
				tex_desc.maxMipmapLevelClamp = 99;
				tex_desc.minMipmapLevelClamp = 0;
				tex_desc.mipmapFilterMode = cudaFilterModePoint;
				tex_desc.borderColor[0] = 1.0f;
				tex_desc.sRGB = 0;

				/* Create the texture object */
				cudaTextureObject_t cuda_tex = 0;
				CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
				m_TextureObjects[tex->textureID] = cuda_tex;
			}
		}
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
		int nObjects = m_Scene->m_Objects.size();
		int meshID = 0;
		std::vector<HitgroupRecord> hitgroupRecords;
		for (int objectID = 0; objectID < nObjects; objectID++)
		{
			auto obj = m_Scene->m_Objects[objectID];
			if (!obj->m_RayTraceRender) continue;

			HitgroupRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(m_HitgroupPGs[0], &rec)); /* For now, all objects use same code */

			/* Textures... */
			if (obj->m_DiffuseTexture.textureID >= 0)
			{
				rec.data.hasDiffuseTexture = true;
				rec.data.diffuseTexture = m_TextureObjects[obj->m_DiffuseTexture.textureID];
			}
			else
			{
				rec.data.hasDiffuseTexture = false;
			}

			if (obj->m_SpecularTexture.textureID >= 0)
			{
				rec.data.hasSpecularTexture = true;
				rec.data.specularTexture = m_TextureObjects[obj->m_SpecularTexture.textureID];
			}
			else
			{
				rec.data.hasSpecularTexture = false;
			}

			if (obj->m_NormalTexture.textureID >= 0)
			{
				rec.data.hasNormalTexture = true;
				rec.data.normalTexture = m_TextureObjects[obj->m_NormalTexture.textureID];
			}
			else
			{
				rec.data.hasNormalTexture = false;
			}

			/* Vertex data */
			rec.data.vertex = (glm::vec3*)m_VertexBuffers[meshID].d_pointer();
			rec.data.index = (glm::ivec3*)m_IndexBuffers[meshID].d_pointer();
			rec.data.normal = (glm::vec3*)m_NormalBuffers[meshID].d_pointer();
			rec.data.texCoord = (glm::vec2*)m_TexCoordBuffers[meshID].d_pointer();
			hitgroupRecords.push_back(rec);

			meshID++;
		}
		m_HitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
		m_SBT.hitgroupRecordBase = m_HitgroupRecordsBuffer.d_pointer();
		m_SBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		m_SBT.hitgroupRecordCount = (int)hitgroupRecords.size();
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
		/* Note: we set the clear/background color here too! */
		m_LaunchParams.clearColor = m_Scene->m_ClearColor;

		m_LastSetCamera = camera;
		m_LaunchParams.camera.position = camera.position;
		m_LaunchParams.camera.direction = glm::normalize(camera.orientation);

		float aspect = m_LaunchParams.frame.size.x / float(m_LaunchParams.frame.size.y);
		float focal_length = glm::length(camera.orientation);
		float h = glm::tan(glm::radians(camera.vfov) / 2.0f);
		float height = 2.0f * h * focal_length;
		float width = height * aspect;
		m_LaunchParams.camera.horizontal = width * glm::normalize(glm::cross(m_LaunchParams.camera.direction, camera.up));
		m_LaunchParams.camera.vertical = height * glm::normalize(glm::cross(m_LaunchParams.camera.horizontal, m_LaunchParams.camera.direction));
	}


	void Optix::Render()
	{
		/* Sanity check: make sure we launch only after first resize is already done */
		if (m_LaunchParams.frame.size.x == 0 || m_LaunchParams.frame.size.y == 0) return;

		m_LaunchParamsBuffer.upload(&m_LaunchParams, 1);
		m_LaunchParams.frameID++;

		OPTIX_CHECK(optixLaunch(
			m_Pipeline,
			m_Stream, 
			m_LaunchParamsBuffer.d_pointer(), 
			m_LaunchParamsBuffer.sizeInBytes, 
			&m_SBT, 
			m_LaunchParams.frame.size.x, 
			m_LaunchParams.frame.size.y, 
			1
		));

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