/*
 * Based on https://github.com/ingowald/optix7course/tree/master
 */

#pragma once

#include "cuda_buffer.h"
#include "launch_params.h"

#include "../camera.h"
#include "../scene.h"
#include "../common_enums.h"

namespace otx
{
	class Optix
	{
	public:
		/* Constructor, performs all setup */
		Optix(std::shared_ptr<Scene> scene);

		/* Resize frame buffer to given resolution */
		void Resize(uint32_t x, uint32_t y);

		/* === Set Methods === */
		void SetCamera(const Camera& camera); /* Set camera for Optix */		
		void SetSamplesPerRender(int nSamples); /* Set the number of samples per pixel per call to Render() */
		void SetMaxDepth(int maxDepth); /* Set the maximum number of ray bounces */
		void SetGammaCorrect(bool correct);
		void SetDenoiserEnabled(bool enabled);
		void SetMaxSampleCount(int nSamples);
		void SetBackgroundRotation(float deg);
		void SetIntegratorType(int integrator);
		void SetSamplerType(int sampler);
		void SetStrataCount(int strata);

		/* === Get Methods === */
		Camera* GetLastSetCamera();
		int GetAccumulatedSampleCount();
		bool GetGammaCorrect();
		int GetSamplesPerRender();
		int GetMaxDepth();
		bool GetDenoiserEnabled();
		int GetMaxSampleCount();
		float GetBackgroundRotation();
		int GetIntegratorType();
		int GetSamplerType();
		int GetStrataCount();


		/* Render one frame */
		void Render();

		/* Download the rendered color buffer */
		void DownloadPixels(uint32_t h_pixels[]);

		/* Set accumulation back to 0 */
		void ResetAccumulation();

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

		/* Setup for callable program(s) we will use */
		void CreateCallablePrograms();

		/* Assembles pipeline of all the programs */
		void CreatePipeline();


		/* Build an acceleration structure for all meshes in m_Meshes */
		OptixTraversableHandle BuildGasAndIas();
		OptixTraversableHandle BuildAccel(OptixBuildInput& buildInput, CUDABuffer& buffer, bool compact = true);

		/* Constructs the shader binding table */
		void BuildSBT();

		/* Upload textures and create cuda texture objects for them */
		void CreateTextures();

		/* Sets up and launches the denoiser (if m_DenoiserEnabled) */
		void LaunchDenoiser();

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
		OptixModule m_PrincipledModule;
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
		std::vector<OptixProgramGroup> m_CallablePGs;
		CUDABuffer m_CallbleRecordsBuffer;
		OptixShaderBindingTable m_SBT = {};

		/* Our launch parameters on the host */
		LaunchParams m_LaunchParams;

		/* The buffer to store LaunchParams on the device*/
		CUDABuffer m_LaunchParamsBuffer;

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

		/* Total accumulated samples per pixel */
		int m_AccumulatedSampleCount = 0;

		/* === Denoiser Components === */
		CUDABuffer m_FBColor; /* The buffer we store the initial, accumulated rendered pixels to, in float4 format */
		CUDABuffer m_FBNormal;
		CUDABuffer m_FBAlbedo;

		/* Output of the denoiser pass, in float4 */
		CUDABuffer m_DenoisedBuffer;

		/* The actual final color buffer to display, in uint8 rgba components */
		CUDABuffer m_FinalColorBuffer;

		OptixDenoiser m_Denoiser = nullptr;
		CUDABuffer m_DenoiserScratch;
		CUDABuffer m_DenoiserState;
		CUDABuffer m_DenoiserIntensity;


		/* === Externally configurable params === */

		int m_IntegratorType = INTEGRATOR_TYPE_PATH;
		int m_SamplerType = SAMPLER_TYPE_STRATIFIED;
		int m_nStrata = 8;

		/* Maximum allowed number of accumulated samples -- set to 0 for unlimited */
		int m_MaxSampleCount = 0;

		/* Samples per pixel per call to render */
		int m_SamplesPerRender = 1;

		/* Maximum number of ray bounces before termination -- set to 0 for russian roulette path termination */
		int m_MaxDepth = 0;

		/* Whether to turn on gamma correction for the final (presented) render */
		bool m_GammaCorrect = true;

		/* Is the denoiser on? We keep it off by default since it really only needs to run once the render is complete. */
		bool m_DenoiserEnabled = false;

		/* Adjust horizontal offset angle of sky texture, locally expressed as degrees */
		float m_BackgroundRotation = 0.0f;
	};


} /* namespace otx */