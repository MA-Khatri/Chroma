/*
 * Based on https://github.com/ingowald/optix7course/tree/master
 */

#pragma once

#include "cuda_buffer.h"
#include "launch_params.h"

#include "imgui.h"
#include "glm/glm.hpp"

namespace otx
{
	class Optix
	{
	public:
		/* Constructor, performs all setup */
		Optix();

		/* Render one frame */
		void Render();

		/* Resize frame buffer to given resolution */
		void Resize(const ImVec2& newSize);

		/* Download the rendered color buffer */
		void DownloadPixels(uint32_t h_pixels[]);


	protected:
		/* Initializes Optix and checks for errors */
		void InitOptix();

		/* Creates and configures an Optix device context */
		void CreateContext();

		/* Creates the module that contains all the programs we will use. */
		void CreateModule();

		/* Setup for raygen program(s) we will use */
		void CreateRaygenPrograms();

		/* Setup for miss program(s) we will use */
		void CreateMissPrograms();

		/* Setup for hitgroup program(s) we will use */
		void CreateHitgroupPrograms();

		/* Assembles pipeline of all the programs */
		void CreatePipeline();

		/* Constructs the shader binding table */
		void BuildSBT();


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

		/* The module that contains our device programs */
		OptixModule m_Module;
		OptixModuleCompileOptions m_ModuleCompileOptions = {};

		/* A vector of all our program(group)s, and the SBT built around them */
		std::vector<OptixProgramGroup> m_RaygenPGs;
		CUDABuffer m_RaygenRecordsBuffer;
		std::vector<OptixProgramGroup> m_MissPGs;
		CUDABuffer m_MissRecordsBuffer;
		std::vector<OptixProgramGroup> m_HitgroupPGs;
		CUDABuffer m_HitgroupRecordsBuffer;
		OptixShaderBindingTable m_SBT = {};

		/* Our launch parameters on the host, and the buffer to store them on the device */
		LaunchParams m_LaunchParams;
		CUDABuffer m_LaunchParamsBuffer;

		/* Our output color buffer */
		CUDABuffer m_ColorBuffer;
	};


} /* namespace otx */