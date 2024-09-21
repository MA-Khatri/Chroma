#include "shader.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include "vulkan_utils.h"


bool print_glsl_code = false;
bool print_spirv_code = false;


namespace VK
{
	std::string ParseShaderFile(std::string filename)
	{
		std::ifstream stream(filename);

		std::string line;
		std::stringstream ss;
		
		while (getline(stream, line))
		{
			ss << line << "\n";
		}

		return ss.str();
	}


	void PreprocessShader(CompilationInfo& info)
	{
		/* Parse the shader using the filename */
		std::string parsedFile = ParseShaderFile(info.fileName);
		info.source = std::vector<char>(parsedFile.begin(), parsedFile.end());

		/* Figure out the kind of shader by looking at the file extension */
		std::string l4c = info.fileName.substr(info.fileName.size() - 4);

		if (l4c == "vert") 
		{
			info.kind = shaderc_vertex_shader;
			info.type = VK_SHADER_STAGE_VERTEX_BIT;
		} 
		else if (l4c == "frag") 
		{
			info.kind = shaderc_fragment_shader;
			info.type = VK_SHADER_STAGE_FRAGMENT_BIT;
		}
		else if (l4c == ".tcs")
		{
			info.kind = shaderc_tess_control_shader;
			info.type = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
		}
		else if (l4c == ".tes")
		{
			info.kind = shaderc_tess_evaluation_shader;
			info.type = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
		}
		else if (l4c == "geom")
		{
			info.kind = shaderc_geometry_shader;
			info.type = VK_SHADER_STAGE_GEOMETRY_BIT;
		}
		else if (l4c == "mesh")
		{
			info.kind = shaderc_mesh_shader;
			info.type = VK_SHADER_STAGE_MESH_BIT_EXT;
		}
		else if (l4c == "comp")
		{
			info.kind = shaderc_compute_shader;
			info.type = VK_SHADER_STAGE_COMPUTE_BIT;
		}
		else
		{
			std::cerr << "File " << info.fileName << " has an unknown shader file extension!" << std::endl;
			exit(-1);
		}

		/* Do the preprocessing */
		shaderc::Compiler compiler;
		shaderc::PreprocessedSourceCompilationResult result = compiler.PreprocessGlsl(info.source.data(), info.source.size(), info.kind, info.fileName.c_str(), info.options);
		if (result.GetCompilationStatus() != shaderc_compilation_status_success)
		{
			std::cerr << result.GetErrorMessage() << std::endl;
			exit(-1);
		}

		/* Copy the result into info for next compilation operation */
		const char* src = result.cbegin();
		size_t newSize = result.cend() - src;
		info.source.resize(newSize);
		memcpy(info.source.data(), src, newSize);

		/* Print output if in debug and set to print glsl code */
#ifdef _DEBUG
		if (print_glsl_code)
		{
			std::string output = { info.source.data(), info.source.data() + info.source.size() };
			std::cout << "---- Preprocessed GLSL source code ----" << std::endl << output << std::endl;
		}
#endif
	}


	std::vector<uint32_t> CompileShader(CompilationInfo& info)
	{
		shaderc::Compiler compiler;

		/* Compile pre-processed GLSL source code to SPIR-V Assembly */
		shaderc::AssemblyCompilationResult result = compiler.CompileGlslToSpvAssembly(info.source.data(), info.source.size(), info.kind, info.fileName.c_str(), info.options);
		if (result.GetCompilationStatus() != shaderc_compilation_status_success)
		{
			std::cerr << result.GetErrorMessage() << std::endl;
			exit(-1);
		}

		/* Copy the result into info for next compilation operation */
		const char* src = result.cbegin();
		size_t newSize = result.cend() - src;
		info.source.resize(newSize);
		memcpy(info.source.data(), src, newSize);

		/* Print output if in debug and set to print spirv code */
#ifdef _DEBUG
		if (print_spirv_code)
		{
			std::string prntOut = { info.source.data(), info.source.data() + info.source.size() };
			std::cout << "---- SPIR-V Assembly code ----" << std::endl << prntOut << std::endl;
		}
#endif


		/* Then compile the assembly down to SPV binary */
		shaderc::SpvCompilationResult result2 = compiler.AssembleToSpv(info.source.data(), info.source.size(), info.options);
		if (result2.GetCompilationStatus() != shaderc_compilation_status_success)
		{
			std::cerr << result2.GetErrorMessage() << std::endl;
			exit(-1);
		}

		/* Copy the result into info for next compilation operation */
		const uint32_t* src2 = result2.cbegin();
		size_t wordCount = result2.cend() - src2;
		std::vector<uint32_t> output(wordCount);
		memcpy(output.data(), src2, wordCount * sizeof(uint32_t));

		return output;
	}


	std::vector<char> ReadShaderFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			std::cerr << "Failed to open file: " << filename << std::endl;
			exit(-1);
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();
		return buffer;
	}


	VkShaderModule CreateShaderModule(const std::vector<char>& code)
	{
		VkResult err;

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		err = vkCreateShaderModule(Device, &createInfo, nullptr, &shaderModule);
		check_vk_result(err);

		return shaderModule;
	}


	ShaderModule CreateShaderModule(const std::string filename)
	{
		VkResult err;

		CompilationInfo info;
		info.fileName = filename;
		info.options.SetOptimizationLevel(shaderc_optimization_level_performance);
		info.options.SetTargetEnvironment(shaderc_target_env_opengl, shaderc_env_version_opengl_4_5);
		info.options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
		PreprocessShader(info);
		std::vector<uint32_t> compiledCode = CompileShader(info);

		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = 4 * compiledCode.size();
		createInfo.pCode = compiledCode.data();

		VkShaderModule shaderModule;
		err = vkCreateShaderModule(Device, &createInfo, nullptr, &shaderModule);
		check_vk_result(err);

		return { shaderModule, info.type };
	}


	std::vector<ShaderModule> CreateShaderModules(std::vector<std::string> filenames)
	{
		std::vector<ShaderModule> modules;

		for (auto& file : filenames)
		{
			modules.push_back(CreateShaderModule(file));
		}

		return modules;
	}


	std::vector<VkPipelineShaderStageCreateInfo> CreateShaderStages(std::vector<ShaderModule> shaderModules)
	{
		/* Create shader stage create info structs */
		std::vector<VkPipelineShaderStageCreateInfo> createInfos;

		for (auto& shaderModule : shaderModules)
		{
			VkPipelineShaderStageCreateInfo shaderStageInfo{};
			shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageInfo.stage = shaderModule.type;
			shaderStageInfo.module = shaderModule.module;
			shaderStageInfo.pName = "main"; /* i.e., entry point */

			createInfos.push_back(shaderStageInfo);
		}

		return createInfos;
	}


	void DestroyShaderModules(std::vector<ShaderModule> shaderModules)
	{
		for (auto shaderModule : shaderModules)
		{
			vkDestroyShaderModule(Device, shaderModule.module, nullptr);
		}
	}
}