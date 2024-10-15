/*
 * Based on https://github.com/ingowald/optix7course/tree/master
 */

#pragma once

#include "imgui.h"

#include "cuda_buffer.h"
#include "launch_params.h"

#include "../camera.h"
#include "../scene.h"

namespace otx
{
	class Optix
	{
	public:
		/* Constructor, performs all setup */
		Optix(std::shared_ptr<Scene> scene);

		/* Resize frame buffer to given resolution */
		void Resize(const ImVec2& newSize);

		/* Set camera for Optix */
		void SetCamera(const Camera& camera);

		/* Set the number of samples per pixel per call to Render() (updates Launch params) */
		void SetSamplesPerRender(int nSamples);

		/* Set the maximum number of ray bounces */
		void SetMaxDepth(int maxDepth);

		Camera* GetLastSetCamera();
		int GetAccumulatedSampleCount();

		/* Render one frame */
		void Render();

		/* Download the rendered color buffer */
		void DownloadPixels(uint32_t h_pixels[]);

	protected:
		/* Initializes Optix and checks for errors */
		void InitOptix();

		/* Creates and configures an Optix device context */
		void CreateContext();

		/* Creates the modules that contain the programs we will use. */
		void CreateModules();

		/* Setup for raygen program(s) we will use */
		void CreateRaygenPrograms();

		/* Setup for miss program(s) we will use */
		void CreateMissPrograms();

		/* Setup for hitgroup program(s) we will use */
		void CreateHitgroupPrograms();

		/* Assembles pipeline of all the programs */
		void CreatePipeline();


		/* Build an acceleration structure for all meshes in m_Meshes */
		OptixTraversableHandle BuildGasAndIas();
		OptixTraversableHandle BuildAccel(OptixBuildInput& buildInput, CUDABuffer& buffer, bool compact = true);

		/* Constructs the shader binding table */
		void BuildSBT();

		/* Upload textures and create cuda texture objects for them */
		void CreateTextures();

	protected:
		/* 
		 * CUDA device context and stream that Optix popeline will run on,
		 * as well as device properties for this device.
		 */
		CUcontext m_CudaContext;
		CUstream m_Stream;
		cudaDeviceProp m_DeviceProps;

		/* The Optix context our pipeline will run in */
		OptixDeviceContext m_OptixContext;

		/* The pipeline we will build */
		OptixPipeline m_Pipeline;
		OptixPipelineCompileOptions m_PipelineCompileOptions = {};
		OptixPipelineLinkOptions m_PipelineLinkOptions = {};

		/* The modules that contain our device programs */
		OptixModule m_RaygenModule;
		OptixModule m_LambertianModule;
		OptixModule m_ConductorModule;
		OptixModule m_DielectricModule;
		OptixModule m_DiffuseLightModule;
		OptixModule m_ShadowModule;
		OptixModule m_MissModule;
		OptixModuleCompileOptions m_ModuleCompileOptions = {};

		/* A vector of all our program(group)s, and the SBT built around them */
		std::vector<OptixProgramGroup> m_RaygenPGs;
		CUDABuffer m_RaygenRecordsBuffer;
		std::vector<OptixProgramGroup> m_MissPGs;
		CUDABuffer m_MissRecordsBuffer;
		std::vector<OptixProgramGroup> m_HitgroupPGs;
		CUDABuffer m_HitgroupRecordsBuffer;
		OptixShaderBindingTable m_SBT = {};

		/* Samples per pixel per call to render */
		int m_SamplesPerRender = 1;

		/* Maximum number of ray bounces before termination */
		int m_MaxDepth = 4;

		/* Our launch parameters on the host */
		LaunchParams m_LaunchParams;

		/* The buffer to store LaunchParams on the device*/
		CUDABuffer m_LaunchParamsBuffer;

		/* Our output color buffer */
		CUDABuffer m_ColorBuffer;

		/* Our accumulated color buffer (where colors are accumulated as floats before conversion to ints) */
		CUDABuffer m_AccumBuffer;

		/* The camera we are using to render */
		Camera m_LastSetCamera;

		/* The scene we are tracing rays against */
		std::shared_ptr<Scene> m_Scene;
		std::vector<std::vector<float>> m_Transforms; /* (static) mesh (pre-)transforms */
		std::vector<CUDABuffer> m_VertexBuffers;
		std::vector<CUDABuffer> m_IndexBuffers;
		std::vector<CUDABuffer> m_NormalBuffers;
		std::vector<CUDABuffer> m_TexCoordBuffers;

		/* Buffers that keep the geometry acceleration structures (per scene object) */
		std::vector<CUDABuffer> m_GASBuffers;
		
		/* Keep the instance of each object... */
		std::vector<OptixInstance> m_Instances;

		/* Buffer that keeps the final, compacted, acceleration structure */
		CUDABuffer m_ASBuffer;

		/* One texture object and pixel array per used texture */
		std::vector<cudaArray_t> m_TextureArrays;
		std::vector<cudaTextureObject_t> m_TextureObjects;

		/* The texture ID of our background image */
		int m_BackgroundTextureID = -1;
	};


} /* namespace otx */