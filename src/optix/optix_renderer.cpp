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

/*
 * Much of the code in this file is based on:
 * https://github.com/ingowald/optix7course
 */

#include "shaders/tone_map.cuh"

namespace otx
{
	float3 ToFloat3(const glm::vec3& v)
	{
		return { v.x, v.y, v.z };
	}

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
		SBTData data;
	};


	Optix::Optix(std::shared_ptr<Scene> scene)
		:m_Scene(scene)
	{
		Debug("[Optix] Initializing Optix...");
		InitOptix();

		Debug("[Optix] Creating context...");
		CreateContext();

		Debug("[Optix] Setting up modules...");
		CreateModules();

		Debug("[Optix] Creating raygen programs...");
		CreateRaygenPrograms();

		Debug("[Optix] Creating miss programs...");
		CreateMissPrograms();

		Debug("[Optix] Creating hitgroup programs...");
		CreateHitgroupPrograms();
		
		Debug("[Optix] Building acceleration structures...");
		m_LaunchParams.traversable = BuildGasAndIas();

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
		fprintf(stderr, "[%2d][%12s]: %s\n", static_cast<int>(level), tag, message);
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


	void Optix::CreateModules()
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
		//m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		//m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
		m_PipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		m_PipelineCompileOptions.usesMotionBlur = false;
		m_PipelineCompileOptions.numPayloadValues = 2;
		m_PipelineCompileOptions.numAttributeValues = 2;
		m_PipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		m_PipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

		m_PipelineLinkOptions.maxTraceDepth = m_MaxDepth; 

		std::vector<std::pair<std::string, OptixModule*>> modules = {
			{ std::string("src/optix/shaders/compiled/raygen.optixir"), &m_RaygenModule },
			{ std::string("src/optix/shaders/compiled/lambertian.optixir"), &m_LambertianModule },
			{ std::string("src/optix/shaders/compiled/conductor.optixir"), &m_ConductorModule },
			{ std::string("src/optix/shaders/compiled/dielectric.optixir"), &m_DielectricModule },
			{ std::string("src/optix/shaders/compiled/diffuse_light.optixir"), &m_DiffuseLightModule },
			{ std::string("src/optix/shaders/compiled/shadow.optixir"), &m_ShadowModule },
			{ std::string("src/optix/shaders/compiled/miss.optixir"), &m_MissModule }
		};

		for (auto& module : modules)
		{
			std::ifstream input(module.first, std::ios::binary);
			std::vector<char> optixirCode(std::istreambuf_iterator<char>(input), {});
			if (optixirCode.empty())
			{
				std::cerr << "Optix::CreateModules(): Failed to load optixir code for file: " << module.first << std::endl;
				exit(-1);
			}

			char log[2048];
			size_t sizeof_log = sizeof(log);
			OPTIX_CHECK(optixModuleCreate(
				m_OptixContext,
				&m_ModuleCompileOptions,
				&m_PipelineCompileOptions,
				optixirCode.data(),
				optixirCode.size(),
				log,
				&sizeof_log,
				module.second
			));

			if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
		}
	}


	void Optix::CreateRaygenPrograms()
	{
		m_RaygenPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pgDesc.raygen.module = m_RaygenModule;
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
		m_MissPGs.resize(RAY_TYPE_COUNT);

		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		pgDesc.miss.module = m_MissModule;

		/* === Radiance rays === */
		pgDesc.miss.entryFunctionName = "__miss__radiance";
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log, 
			&sizeof_log, 
			&m_MissPGs[RAY_TYPE_RADIANCE]
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;

		/* === Shadow rays === */
		pgDesc.miss.module = m_ShadowModule;
		pgDesc.miss.entryFunctionName = "__miss__shadow";
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext,
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&m_MissPGs[RAY_TYPE_SHADOW]
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
	}


	void Optix::CreateHitgroupPrograms()
	{
		/* 
		 * We do -1 to remove RAY_TYPE_RADIANCE since we have no radiance-ray specific hitgroup 
		 * programs but still need to add hitgroup programs for other ray types.
		 */
		m_HitgroupPGs.resize(MATERIAL_TYPE_COUNT + RAY_TYPE_COUNT - 1);

		char log[2048];
		size_t sizeof_log = sizeof(log);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc = {};
		pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		pgDesc.hitgroup.moduleAH = m_LambertianModule;
		pgDesc.hitgroup.moduleCH = m_LambertianModule;

		/* === Radiance rays === */
		/* Lambertian */
		pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
		pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext, 
			&pgDesc, 
			1, 
			&pgOptions, 
			log, 
			&sizeof_log, 
			&m_HitgroupPGs[MATERIAL_TYPE_LAMBERTIAN]
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;

		/* Conductor */
		pgDesc.hitgroup.moduleAH = m_ConductorModule;
		pgDesc.hitgroup.moduleCH = m_ConductorModule;
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext,
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&m_HitgroupPGs[MATERIAL_TYPE_CONDUCTOR]
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;

		/* Dielectric */
		pgDesc.hitgroup.moduleAH = m_DielectricModule;
		pgDesc.hitgroup.moduleCH = m_DielectricModule;
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext,
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&m_HitgroupPGs[MATERIAL_TYPE_DIELECTRIC]
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;

		/* Diffuse Light */
		pgDesc.hitgroup.moduleAH = m_DiffuseLightModule;
		pgDesc.hitgroup.moduleCH = m_DiffuseLightModule;
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext,
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&m_HitgroupPGs[MATERIAL_TYPE_DIFFUSE_LIGHT]
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;

		// TODO, more...

		/* === Shadow rays === */
		pgDesc.hitgroup.moduleAH = m_ShadowModule;
		pgDesc.hitgroup.moduleCH = m_ShadowModule;
		pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
		pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
		OPTIX_CHECK(optixProgramGroupCreate(
			m_OptixContext,
			&pgDesc,
			1,
			&pgOptions,
			log,
			&sizeof_log,
			&m_HitgroupPGs[MATERIAL_TYPE_COUNT + RAY_TYPE_SHADOW - 1]
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
			static_cast<int>(programGroups.size()), 
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
			2		    /* [in] The maximum depth of a traversable graph passed to trace */
		));
		if (sizeof_log > 1 && debug_mode) std::cout << "Log: " << log << std::endl;
	}


	OptixTraversableHandle Optix::BuildAccel(OptixBuildInput& buildInput, CUDABuffer& buffer, bool compact /* = true*/)
	{
		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 0; /* No motion blur */
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes asBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			m_OptixContext,
			&accelOptions,
			&buildInput,
			1,
			&asBufferSizes
		));

		/* === Prepare compaction === */
		CUDABuffer compactedSizeBuffer;
		OptixAccelEmitDesc emitDesc;

		if (compact)
		{
			compactedSizeBuffer.alloc(sizeof(uint64_t));

			emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
			emitDesc.result = compactedSizeBuffer.d_pointer();
		}

		/* === Execute build === */
		OptixTraversableHandle asHandle{ 0 };

		CUDABuffer tempBuffer;
		tempBuffer.alloc(asBufferSizes.tempSizeInBytes);

		CUDABuffer outputBuffer;
		if (compact) outputBuffer.alloc(asBufferSizes.outputSizeInBytes);
		else buffer.alloc(asBufferSizes.outputSizeInBytes);

		OPTIX_CHECK(optixAccelBuild(
			m_OptixContext,
			0, /* stream */
			&accelOptions,
			&buildInput,
			1,
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,
			compact ? outputBuffer.d_pointer() : buffer.d_pointer(),
			compact ? outputBuffer.sizeInBytes : buffer.sizeInBytes,
			&asHandle,
			compact ? &emitDesc : nullptr,
			compact ? 1 : 0
		));
		CUDA_SYNC_CHECK();

		/* === Perform compaction === */
		if (compact)
		{
			uint64_t compactedSize;
			compactedSizeBuffer.download(&compactedSize, 1);

			buffer.alloc(compactedSize);
			OPTIX_CHECK(optixAccelCompact(
				m_OptixContext,
				0,
				asHandle,
				buffer.d_pointer(),
				buffer.sizeInBytes,
				&asHandle
			));
			CUDA_SYNC_CHECK();
		}

		/* === Clean up === */
		if (compact)
		{
			outputBuffer.free();
			compactedSizeBuffer.free();
		}
		tempBuffer.free();

		return asHandle;
	}


	OptixTraversableHandle Optix::BuildGasAndIas()
	{
		auto& objects = m_Scene->m_RayTraceObjects;
		int nObjects = static_cast<int>(objects.size());

		/* Extract transforms from scene objects */
		m_Transforms.reserve(nObjects);
		for (auto& object : objects)
		{
			std::vector<float> transform;
			transform.reserve(12);
			const glm::mat4& t = object->m_ModelMatrix;
			for (int row = 0; row < 3; row++)
			{
				for (int col = 0; col < 4; col++)
				{
					transform.emplace_back(t[col][row]);
				}
			}
			m_Transforms.emplace_back(transform);
		}

		/* Get ready to store mesh data on device... */
		m_VertexBuffers.resize(nObjects);
		m_IndexBuffers.resize(nObjects);
		m_NormalBuffers.resize(nObjects);
		m_TexCoordBuffers.resize(nObjects);

		m_GASBuffers.resize(nObjects);
		m_Instances.resize(nObjects);

		/* ==================================== */
		/* === GAS and IAS per scene object === */
		/* ==================================== */
		for (int objectID = 0; objectID < nObjects; objectID++)
		{
			/* === GAS Setup === */
			OptixBuildInput triangleInput;
			CUdeviceptr d_vertices;
			CUdeviceptr d_indices;
			uint32_t triangleInputFlags;

			/* Upload the mesh to the device */
			Mesh& mesh = *(objects[objectID]->m_Mesh);
			m_VertexBuffers[objectID].alloc_and_upload(mesh.posns);
			m_IndexBuffers[objectID].alloc_and_upload(mesh.ivecIndices);
			m_NormalBuffers[objectID].alloc_and_upload(mesh.normals);
			m_TexCoordBuffers[objectID].alloc_and_upload(mesh.texCoords);

			triangleInput = {};
			triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			/* Create local variables to store pointers to the device pointers */
			d_vertices = m_VertexBuffers[objectID].d_pointer();
			d_indices = m_IndexBuffers[objectID].d_pointer();

			/* Set up format for reading vertex and index data */
			triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
			triangleInput.triangleArray.numVertices = static_cast<int>(mesh.posns.size());
			triangleInput.triangleArray.vertexBuffers = &d_vertices;

			triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput.triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
			triangleInput.triangleArray.numIndexTriplets = static_cast<int>(mesh.ivecIndices.size());
			triangleInput.triangleArray.indexBuffer = d_indices;

			triangleInputFlags = 0;

			/* For now, we only have one SBT entry and no per-primitive materials */
			triangleInput.triangleArray.flags = &triangleInputFlags;
			triangleInput.triangleArray.numSbtRecords = 1;
			triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
			triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
			triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

			/* === Build GAS === */
			OptixTraversableHandle gasHandle = BuildAccel(triangleInput, m_GASBuffers[objectID]);

			/* === IAS Setup === */
			OptixInstance instance = {};
			memcpy(instance.transform, m_Transforms[objectID].data(), sizeof(float) * 12); /* Copy over the object's transform */
			instance.instanceId = objectID;
			instance.visibilityMask = 255;
			instance.sbtOffset = objectID * RAY_TYPE_COUNT;			
			instance.flags = OPTIX_INSTANCE_FLAG_NONE;
			instance.traversableHandle = gasHandle;

			m_Instances[objectID] = instance;
		}

		/* === Top level IAS Setup === */
		void* d_instances;
		cudaMalloc(&d_instances, m_Instances.size() * sizeof(m_Instances[0]));
		cudaMemcpy(d_instances, m_Instances.data(), m_Instances.size() * sizeof(m_Instances[0]), cudaMemcpyHostToDevice);

		OptixBuildInput topIasInput = {};
		topIasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		topIasInput.instanceArray.instances = (CUdeviceptr)d_instances;
		topIasInput.instanceArray.numInstances = (int)m_Instances.size();

		OptixTraversableHandle asHandle = BuildAccel(topIasInput, m_ASBuffer, false);

		return asHandle;
	}


	void Optix::CreateTextures()
	{
		/* Get the texture count */
		int nTextures = 0;
		for (auto obj : m_Scene->m_RayTraceObjects)
		{
			auto mat = obj->m_Material;
			if (mat->m_DiffuseTexture.pixels.size() > 0) nTextures++;
			if (mat->m_SpecularTexture.pixels.size() > 0) nTextures++;
			if (mat->m_NormalTexture.pixels.size() > 0) nTextures++;
		}

		if (m_Scene->m_BackgroundTexture.pixels.size() > 0)
		{
			nTextures++;
		}

		m_TextureArrays.resize(nTextures);
		m_TextureObjects.resize(nTextures);

		int textureID = 0;
		for (auto obj : m_Scene->m_RayTraceObjects)
		{
			auto mat = obj->m_Material;

			/* Get all textures for this object */
			std::vector<Texture<uint8_t>*> textures;
			textures.reserve(3);

			Texture<uint8_t>* diffuse = &(mat->m_DiffuseTexture);
			if (diffuse->pixels.size() > 0) { diffuse->textureID = textureID; textureID++; textures.emplace_back(diffuse); }

			Texture<uint8_t>* specular = &(mat->m_SpecularTexture);
			if (specular->pixels.size() > 0) { specular->textureID = textureID; textureID++; textures.emplace_back(specular); }

			Texture<uint8_t>* normal = &(mat->m_NormalTexture);
			if (normal->pixels.size() > 0) { normal->textureID = textureID; textureID++; textures.emplace_back(normal); }

			/* Create CUDA resources for each texture */
			for (Texture<uint8_t>* tex : textures)
			{
				int32_t width = tex->resolution.x;
				int32_t height = tex->resolution.y;
				int32_t numComponents = tex->resolution.z;
				int32_t pitch = width * numComponents * sizeof(uint8_t);
				cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

				cudaArray_t& pixelArray = m_TextureArrays[tex->textureID];
				CUDA_CHECK(MallocArray(&pixelArray, &channel_desc, width, height));
				CUDA_CHECK(Memcpy2DToArray(pixelArray, 0, 0, tex->pixels.data(), pitch, pitch, height, cudaMemcpyHostToDevice));

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

		/* Load background texture if there is one */
		if (m_Scene->m_BackgroundTexture.pixels.size() > 0)
		{
			auto& tex = m_Scene->m_BackgroundTexture;
			tex.textureID = textureID;

			int32_t width = tex.resolution.x;
			int32_t height = tex.resolution.y;
			int32_t numComponents = tex.resolution.z;
			int32_t pitch = width * numComponents * sizeof(float);
			cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();

			cudaArray_t& pixelArray = m_TextureArrays[tex.textureID];
			CUDA_CHECK(MallocArray(&pixelArray, &channel_desc, width, height));
			CUDA_CHECK(Memcpy2DToArray(pixelArray, 0, 0, tex.pixels.data(), pitch, pitch, height, cudaMemcpyHostToDevice));

			cudaResourceDesc res_desc = {};
			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = pixelArray;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeElementType;

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
			m_TextureObjects[tex.textureID] = cuda_tex;
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
		m_SBT.missRecordCount = static_cast<int>(missRecords.size());

		/* Build hitgroup records */
		int nObjects = static_cast<int>(m_Scene->m_RayTraceObjects.size());
		std::vector<HitgroupRecord> hitgroupRecords;
		for (int objectID = 0; objectID < nObjects; objectID++)
		{
			for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
			{
				auto obj = m_Scene->m_RayTraceObjects[objectID];
				auto mat = obj->m_Material;
				HitgroupRecord rec;

				/* RADIANCE rays only */
				if (rayID == RAY_TYPE_RADIANCE)
				{
					/* Assign the material type (program) here */
					OPTIX_CHECK(optixSbtRecordPackHeader(m_HitgroupPGs[mat->m_RTMaterialType], &rec));

					/* Textures... */
					if (mat->m_DiffuseTexture.textureID >= 0)
					{
						rec.data.hasDiffuseTexture = true;
						rec.data.diffuseTexture = m_TextureObjects[mat->m_DiffuseTexture.textureID];
					}
					else
					{
						rec.data.hasDiffuseTexture = false;
					}

					if (mat->m_SpecularTexture.textureID >= 0)
					{
						rec.data.hasSpecularTexture = true;
						rec.data.specularTexture = m_TextureObjects[mat->m_SpecularTexture.textureID];
					}
					else
					{
						rec.data.hasSpecularTexture = false;
					}

					if (mat->m_NormalTexture.textureID >= 0)
					{
						rec.data.hasNormalTexture = true;
						rec.data.normalTexture = m_TextureObjects[mat->m_NormalTexture.textureID];
					}
					else
					{
						rec.data.hasNormalTexture = false;
					}

					rec.data.texCoord = (float2*)m_TexCoordBuffers[objectID].d_pointer();
					rec.data.normal = (float3*)m_NormalBuffers[objectID].d_pointer();
					
					/* Set remaining material properties */
					rec.data.roughness = mat->m_Roughness;
					rec.data.etaIn = mat->m_EtaIn;
					rec.data.etaOut = mat->m_EtaOut;
					rec.data.reflectionColor = ToFloat3(mat->m_ReflectionColor);
					rec.data.refractionColor = ToFloat3(mat->m_RefractionColor);
					rec.data.extinction = ToFloat3(mat->m_Extinction);
				}
				/* Shadow rays only */
				else if (rayID == RAY_TYPE_SHADOW)
				{
					OPTIX_CHECK(optixSbtRecordPackHeader(m_HitgroupPGs[rayID + MATERIAL_TYPE_COUNT - 1], &rec));
				}
				else
				{
					std::cerr << "Optix::BuildSBT(): WARNING! Invalid rayID! Skipping..." << std::endl;
					continue;
				}

				/* Common to all raytypes -- i.e., mesh data */
				rec.data.position = (float3*)m_VertexBuffers[objectID].d_pointer();
				rec.data.index = (int3*)m_IndexBuffers[objectID].d_pointer();
				
				hitgroupRecords.push_back(rec);
			}
		}

		if (nObjects > 0)
		{
			m_HitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
			m_SBT.hitgroupRecordBase = m_HitgroupRecordsBuffer.d_pointer();
			m_SBT.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
			m_SBT.hitgroupRecordCount = static_cast<int>(hitgroupRecords.size());
		}
	}


	void Optix::Resize(uint32_t x, uint32_t y)
	{
		if (m_DenoiserOn)
		{
			OPTIX_CHECK(optixDenoiserDestroy(m_Denoiser));
		}

		/* === Create the denoiser === */
		OptixDenoiserOptions denoiserOptions = {};
		OPTIX_CHECK(optixDenoiserCreate(m_OptixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &m_Denoiser));

		/* Compute and allocate memory resources for the denoiser */
		OptixDenoiserSizes denoiserReturnSizes;
		OPTIX_CHECK(optixDenoiserComputeMemoryResources(
			m_Denoiser,
			x, y,
			&denoiserReturnSizes
		));

		m_DenoiserScratch.resize(std::max(denoiserReturnSizes.withoutOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
		m_DenoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

		/* Resize our CUDA framebuffer */
		size_t fsize = static_cast<size_t>(x) * static_cast<size_t>(y) * sizeof(float4);
		m_DenoisedBuffer.resize(fsize);
		m_FBColor.resize(fsize);
		m_FBNormal.resize(fsize);
		m_FBAlbedo.resize(fsize);
		m_FinalColorBuffer.resize(static_cast<size_t>(x) * static_cast<size_t>(y) * sizeof(uint32_t));

		/* Update our launch parameters */
		m_LaunchParams.frame.size.x = static_cast<int>(x);
		m_LaunchParams.frame.size.y = static_cast<int>(y);
		m_LaunchParams.frame.colorBuffer = (float4*)m_FBColor.d_pointer();
		m_LaunchParams.frame.normalBuffer = (float4*)m_FBNormal.d_pointer();
		m_LaunchParams.frame.albedoBuffer = (float4*)m_FBAlbedo.d_pointer();
		m_LaunchParams.frame.accumID = 0;

		OPTIX_CHECK(optixDenoiserSetup(
			m_Denoiser,
			0,
			x, y,
			m_DenoiserState.d_pointer(),
			m_DenoiserState.sizeInBytes,
			m_DenoiserScratch.d_pointer(),
			m_DenoiserScratch.sizeInBytes
		));

		/* Reset the camera because our aspect ratio may have changed */
		SetCamera(m_LastSetCamera);
	}


	void Optix::SetCamera(const Camera& camera)
	{
		/* === Update camera === */
		m_LastSetCamera = camera;
		m_LaunchParams.camera.position = ToFloat3(camera.m_Position);
		m_LaunchParams.camera.direction = ToFloat3(glm::normalize(camera.m_Orientation));

		if (camera.m_ProjectionMode == PROJECTION_MODE_PERSPECTIVE)
		{
			float aspect = m_LaunchParams.frame.size.x / float(m_LaunchParams.frame.size.y);
			float focal_length = glm::length(camera.m_Orientation);
			float h = glm::tan(glm::radians(camera.m_VFoV) / 2.0f);
			float height = 2.0f * h * focal_length;
			float width = height * aspect;
			m_LaunchParams.camera.horizontal = width * normalize(cross(m_LaunchParams.camera.direction, ToFloat3(camera.m_Up)));
			m_LaunchParams.camera.vertical = height * normalize(cross(m_LaunchParams.camera.horizontal, m_LaunchParams.camera.direction));
			m_LaunchParams.camera.projectionMode = PROJECTION_MODE_PERSPECTIVE;
		}
		else if (camera.m_ProjectionMode == PROJECTION_MODE_ORTHOGRAPHIC)
		{
			m_LaunchParams.camera.horizontal = m_LaunchParams.frame.size.x * camera.m_OrthoScale * normalize(cross(m_LaunchParams.camera.direction, ToFloat3(camera.m_Up)));
			m_LaunchParams.camera.vertical = m_LaunchParams.frame.size.y * camera.m_OrthoScale * normalize(cross(m_LaunchParams.camera.horizontal, m_LaunchParams.camera.direction));
			m_LaunchParams.camera.projectionMode = PROJECTION_MODE_ORTHOGRAPHIC;
		}
		else if (camera.m_ProjectionMode == PROJECTION_MODE_THIN_LENS)
		{
			float aspect = m_LaunchParams.frame.size.x / float(m_LaunchParams.frame.size.y);
			float h = glm::tan(glm::radians(camera.m_VFoV) / 2.0f);
			float height = 2.0f * h * camera.m_FocusDistance;
			float width = height * aspect;

			float3 u = normalize(cross(m_LaunchParams.camera.direction, ToFloat3(camera.m_Up)));
			float3 v = normalize(cross(m_LaunchParams.camera.horizontal, m_LaunchParams.camera.direction));
			
			m_LaunchParams.camera.horizontal = width * u;
			m_LaunchParams.camera.vertical = height * v;
			m_LaunchParams.camera.direction *= camera.m_FocusDistance;
			m_LaunchParams.camera.projectionMode = PROJECTION_MODE_THIN_LENS;

			/* Determine the defocus disk basis vectors */
			float defocusRadius = camera.m_FocusDistance * glm::tan(glm::radians(camera.m_DefocusAngle / 2.0f));
			m_LaunchParams.camera.defocusDiskU = defocusRadius * u;
			m_LaunchParams.camera.defocusDiskV = defocusRadius * v;
		}

		/* Reset accumulation */
		m_LaunchParams.frame.accumID = 0;
		m_AccumulatedSampleCount = 0;
	}


	void Optix::SetSamplesPerRender(int nSamples)
	{
		m_SamplesPerRender = nSamples;
	}


	void Optix::SetMaxDepth(int maxDepth)
	{
		m_MaxDepth = maxDepth;

		/* Reset accumulation */
		m_LaunchParams.frame.accumID = 0;
	}

	void Optix::SetGammaCorrect(bool correct)
	{
		m_GammaCorrect = correct;
	}

	Camera* Optix::GetLastSetCamera()
	{
		return &m_LastSetCamera;
	}

	int Optix::GetAccumulatedSampleCount()
	{
		return m_AccumulatedSampleCount;
	}

	bool Optix::GetGammaCorrect()
	{
		return m_GammaCorrect;
	}

	int Optix::GetSamplesPerRender()
	{
		return m_SamplesPerRender;
	}

	int Optix::GetMaxDepth()
	{
		return m_MaxDepth;
	}

	void Optix::Render()
	{
		/* Sanity check: make sure we launch only after first resize is already done */
		if (m_LaunchParams.frame.size.x == 0 || m_LaunchParams.frame.size.y == 0) return;

		/* === Update launch Params here === */
		m_LaunchParams.frame.samples = m_SamplesPerRender;
		m_LaunchParams.maxDepth = m_MaxDepth;
		m_LaunchParams.cutoffColor = make_float3(0.0f);
		m_LaunchParams.gammaCorrect = m_GammaCorrect;

		/* Background settings */
		m_LaunchParams.backgroundMode = m_Scene->m_BackgroundMode;
		m_LaunchParams.clearColor = ToFloat3(m_Scene->m_ClearColor);
		m_LaunchParams.gradientBottom = ToFloat3(m_Scene->m_GradientBottom);
		m_LaunchParams.gradientTop = ToFloat3(m_Scene->m_GradientTop);

		int backgroundID = m_Scene->m_BackgroundTexture.textureID;
		if (backgroundID >= 0) m_LaunchParams.backgroundTexture = m_TextureObjects[backgroundID];
		m_LaunchParams.gammaCorrect = m_GammaCorrect;

		/* Upload launch params */
		m_LaunchParamsBuffer.upload(&m_LaunchParams, 1);
		m_LaunchParams.frame.accumID++; /* Must increment *after* upload */

		/* === OptixLaunch === */
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

		m_AccumulatedSampleCount += m_SamplesPerRender;


		/* === Denoiser Setup === */
		m_DenoiserIntensity.resize(sizeof(float));
		
		OptixDenoiserParams denoiserParams = {};
		if (m_DenoiserIntensity.sizeInBytes != sizeof(float))
		{
			m_DenoiserIntensity.alloc(sizeof(float));
		}
		denoiserParams.hdrIntensity = m_DenoiserIntensity.d_pointer();
		denoiserParams.blendFactor = 1.0f / (m_LaunchParams.frame.accumID);

		OptixImage2D inputLayer[3];
		inputLayer[0].data = m_FBColor.d_pointer();
		inputLayer[0].width = m_LaunchParams.frame.size.x; /* Width in pixels */
		inputLayer[0].height = m_LaunchParams.frame.size.y; /* Height in pixels */
		inputLayer[0].rowStrideInBytes = m_LaunchParams.frame.size.x * sizeof(float4);
		inputLayer[0].pixelStrideInBytes = sizeof(float4);
		inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

		inputLayer[1].data = m_FBAlbedo.d_pointer();
		inputLayer[1].width = m_LaunchParams.frame.size.x;
		inputLayer[1].height = m_LaunchParams.frame.size.y;
		inputLayer[1].rowStrideInBytes = m_LaunchParams.frame.size.x * sizeof(float4);
		inputLayer[1].pixelStrideInBytes = sizeof(float4);
		inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

		inputLayer[2].data = m_FBNormal.d_pointer();
		inputLayer[2].width = m_LaunchParams.frame.size.x;
		inputLayer[2].height = m_LaunchParams.frame.size.y;
		inputLayer[2].rowStrideInBytes = m_LaunchParams.frame.size.x * sizeof(float4);
		inputLayer[2].pixelStrideInBytes = sizeof(float4);
		inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

		OptixImage2D outputLayer;
		outputLayer.data = m_DenoisedBuffer.d_pointer();
		outputLayer.width = m_LaunchParams.frame.size.x;
		outputLayer.height = m_LaunchParams.frame.size.y;
		outputLayer.rowStrideInBytes = m_LaunchParams.frame.size.x * sizeof(float4);
		outputLayer.pixelStrideInBytes = sizeof(float4);
		outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

		if (m_DenoiserOn)
		{
			OPTIX_CHECK(optixDenoiserComputeIntensity(
				m_Denoiser, 
				0, /* stream */
				&inputLayer[0], 
				(CUdeviceptr)m_DenoiserIntensity.d_pointer(),
				(CUdeviceptr)m_DenoiserScratch.d_pointer(),
				m_DenoiserScratch.sizeInBytes
			));

			OptixDenoiserGuideLayer denoiserGuideLayer = {};
			denoiserGuideLayer.albedo = inputLayer[1];
			denoiserGuideLayer.normal = inputLayer[2];

			OptixDenoiserLayer denoiserLayer = {};
			denoiserLayer.input = inputLayer[0];
			denoiserLayer.output = outputLayer;

			OPTIX_CHECK(optixDenoiserInvoke(
				m_Denoiser,
				0, /* stream */
				&denoiserParams,
				m_DenoiserState.d_pointer(),
				m_DenoiserState.sizeInBytes,
				&denoiserGuideLayer,
				&denoiserLayer, 1,
				0, /* input offset x */
				0, /* input offset y */
				m_DenoiserScratch.d_pointer(),
				m_DenoiserScratch.sizeInBytes
			));
		}
		else
		{
			cudaMemcpy(
				(void*)outputLayer.data, 
				(void*)inputLayer[0].data,
				outputLayer.width * outputLayer.height * sizeof(float4),
				cudaMemcpyDeviceToDevice
			);
		}
		ComputeFinalPixelColors(m_LaunchParams.frame.size, (uint32_t*)m_FinalColorBuffer.d_pointer(), (float4*)m_DenoisedBuffer.d_pointer(), m_GammaCorrect);

		/*
		 * Make sure frame is rendered before we display. BUT -- Vulkan does not know when this is finished!
		 * For higher performance, we should use streams and do double-buffering
		 */
		CUDA_SYNC_CHECK();
	}


	void Optix::DownloadPixels(uint32_t h_pixels[])
	{
		m_FinalColorBuffer.download(h_pixels, m_LaunchParams.frame.size.x * m_LaunchParams.frame.size.y);
	}
}