#pragma once

#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.h>

#include <vector>


namespace VK
{
	/* Compilation info used in preprocess and compilation */
	struct CompilationInfo
	{
		/* Filename of the original file, used in Debug messages */
		std::string fileName;

		/* Shader kind, i.e. type of shader to be produced */
		shaderc_shader_kind kind;

		/* Final type of shader used in shader stage creation */
		VkShaderStageFlagBits type;

		/* Shader source code */
		std::vector<char> source;

		/* Compilation options */
		shaderc::CompileOptions options;
	};


	/* Stores VkShaderModule and its corresponding type */
	struct ShaderModule
	{
		VkShaderModule module;
		VkShaderStageFlagBits type;
	};


	std::string ParseShaderFile(std::string filename);


	/* 
	 * Takes in the info struct with at least the filename and then parses, 
	 * preprocesses, and determines the kind using the file extension.
	 */
	void PreprocessShader(CompilationInfo& info);


	/*
	 * Compiles the pre-processed GLSL code from PreprocessShader and 
	 * compiles to SPIR-V assembly, then to SPIR-V binary.
	 * Returns the compoled binary shader code.
	 */
	std::vector<uint32_t> CompileShader(CompilationInfo& info);


	/* Read in pre-compiled SPIR-V files */
	std::vector<char> ReadShaderFile(const std::string& filename);


	/* Create shader module from pre-compiled source code */
	VkShaderModule CreateShaderModule(const std::vector<char>& code);


	/* 
	 * Create shader module using the filename and compiling at run time.
	 * Note: file extension must end in either:
	 * .vert, .frag, .tcs, .tes, .geom, .mesh, .comp
	 * to correctly determine the shader type.
	 * Note: returns ShaderModule struct with shader module and type (kind)
	 */
	ShaderModule CreateShaderModule(const std::string filename);


	std::vector<ShaderModule> CreateShaderModules(std::vector<std::string> filenames);

	std::vector<VkPipelineShaderStageCreateInfo> CreateShaderStages(std::vector<ShaderModule> shaderModules);
	
	void DestroyShaderModules(std::vector<ShaderModule> shaderModules);
}
