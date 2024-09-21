#pragma once

#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.h>

#include <vector>


namespace VK
{
	struct CompilationInfo;


	std::string ParseShaderFile(std::string filename);


	/* 
	 * Takes in the info struct with at least the filename and then parses, 
	 * preprocesses, and determines the kind using the file extension.
	 */
	void PreprocessShader(CompilationInfo& info);


	/*
	 * Compiles the pre-processed GLSL code from PreprocessShader and compiles to SPIR-V assembly 
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
	 */
	VkShaderModule CreateShaderModule(const std::string filename);
}
