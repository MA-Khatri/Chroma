#pragma once

#include <iostream>
#include <optional>
#include <functional>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <vulkan/vulkan.h>

#include "../mesh.h"
#include "../object.h"


//#define APP_USE_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define APP_USE_VULKAN_DEBUG_REPORT
#endif


namespace vk
{
	/* === Namespace Globals === */
	
	extern VkInstance Instance;
	extern VkPhysicalDevice PhysicalDevice;
	extern VkDevice Device;
	extern VkDescriptorPool DescriptorPool;
	extern VkPipelineCache PipelineCache;
	extern VkCommandPool TransferCommandPool;
	extern VkCommandPool GraphicsCommandPool;

	extern ImGui_ImplVulkanH_Window MainWindowData;
	extern uint32_t MinImageCount; /* >= 2 */
	extern uint32_t ImageCount; /* >= MinImageCount */
	extern bool SwapChainRebuild;

	extern VkSampleCountFlagBits MaxMSAASamples;

	extern uint32_t GraphicsQueueFamily;
	extern uint32_t ComputeQueueFamily;
	extern uint32_t TransferQueueFamily;

	extern VkQueue GraphicsQueue;
	extern VkQueue ComputeQueue;
	extern VkQueue TransferQueue;

	extern VkDebugReportCallbackEXT DebugReport;
	extern VkAllocationCallbacks* Allocator;

	/* Per-frame-in-flight */
	extern std::vector<std::vector<VkCommandBuffer>> AllocatedGraphicsCommandBuffers;
	extern std::vector<std::vector<std::function<void()>>> ResourceFreeQueue;

	/*
	Unlike g_MainWindowData.FrameIndex, this is not the the swapchain image index
	and is always guaranteed to increase (eg. 0, 1, 2, 0, 1, 2)
	*/
	extern uint32_t CurrentFrameIndex;


	/* === Error Handling Utilities === */
	
	void glfw_error_callback(int error, const char* description);

	void check_vk_result(VkResult err);

#ifdef APP_USE_VULKAN_DEBUG_REPORT
	VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(
		VkDebugReportFlagsEXT flags,
		VkDebugReportObjectTypeEXT objectType,
		uint64_t object,
		size_t location,
		int32_t messageCode,
		const char* pLayerPrefix,
		const char* pMessage,
		void* pUserData
	);
#endif /* APP_USE_VULKAN_DEBUG_REPORT */


	
	/* === Vulkan Utility Functions === */
	
	bool IsExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension);

	VkCommandBuffer GetGraphicsCommandBuffer();
	void FlushGraphicsCommandBuffer(VkCommandBuffer commandBuffer);

	VkCommandBuffer GetTransferCommandBuffer();
	void FlushTransferCommandBuffer(VkCommandBuffer commandBuffer);

	void SubmitResourceFree(std::function<void()>&& func);

	uint32_t GetVulkanMemoryType(VkMemoryPropertyFlags properties, uint32_t type_bits);

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat FindDepthFormat();
	bool HasStencilComponent(VkFormat format);
	
	/* === Vulkan Setup Functions === */
	
	void SetupVulkan(ImVector<const char*> instance_extensions);
	void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height);

	void CreateInstance(ImVector<const char*> instance_extensions);
	void SelectPhysicalDevice();
	void GetQueueFamilies();
	void CreateLogicalDevice();
	void CreateDescriptorPool();
	void CreateTransientCommandPool(uint32_t queueFamily, VkCommandPool& commandPool);

	VkSampleCountFlagBits GetMaxUsableSampleCount();

	void CleanupVulkan();
	void CleanupVulkanWindow();

	
	/* === ImGui Utility Functions === */
	
	void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data);
	void FramePresent(ImGui_ImplVulkanH_Window* wd);

	
	/* === Layer Utility Functions === */
	
	/* For creating viewport images */
	void CreateViewportImage(ImVec2 extent, VkImage& image, VkDeviceMemory& memory);
	void CreateViewportImages(uint32_t count, ImVec2 extent, std::vector<VkImage>& images, std::vector<VkDeviceMemory>& memory);

	void CreateImageView(VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels, VkImage& image, VkImageView& imageView);
	void CreateViewportImageViews(std::vector<VkImage>& images, std::vector<VkImageView>& views);

	void CreateRenderPass(VkSampleCountFlagBits msaaSamples, VkRenderPass& renderPass);

	void CreateGraphicsPipeline(std::vector<std::string> shaderFiles, ImVec2 extent, VkSampleCountFlagBits msaaSamples, VkPrimitiveTopology topology, const VkRenderPass& renderPass, const VkDescriptorSetLayout& descriptorSetLayout, VkPipelineLayout& layout, VkPipeline& pipeline);
	VkPipeline CreateGraphicsPipeline(std::vector<std::string> shaderFiles, ImVec2 extent, VkSampleCountFlagBits msaaSamples, VkPrimitiveTopology topology, const VkRenderPass& renderPass, const VkDescriptorSetLayout& descriptorSetLayout, VkPipelineLayout& layout);

	void CreateFrameBuffer(std::vector<VkImageView> attachments, VkRenderPass& renderPass, ImVec2 extent, VkFramebuffer& framebuffer);
	void CreateFrameBuffers(std::vector<VkImageView> attachments, VkRenderPass& renderPass, ImVec2 extent, uint32_t count, std::vector<VkFramebuffer>& framebuffers);

	void CreateViewportSampler(VkSampler* sampler);

	void CreateColorResources(uint32_t width, uint32_t height, VkSampleCountFlagBits msaaSamples, VkImage& colorImage, VkDeviceMemory& colorImageMemory, VkImageView& colorImageView);

	void CreateDepthResources(uint32_t width, uint32_t height, VkSampleCountFlagBits msaaSamples, VkImage& depthImage, VkDeviceMemory& depthImageMemory, VkImageView& depthImageView);

	/* === Buffers === */
	
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	
	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

	void CreateVertexBuffer(const std::vector<Vertex> vertices, VkBuffer& vertexBuffer, VkDeviceMemory& vertexBufferMemory);
	void CreateIndexBuffer(const std::vector<uint32_t> indices, VkBuffer& indexBuffer, VkDeviceMemory& indexBufferMemory);

	void CreateDescriptorSetLayout(std::vector<VkDescriptorSetLayoutBinding>& layoutBindings, VkDescriptorSetLayout& descriptorSetLayout);
	void CreateUniformBuffer(VkDeviceSize bufferSize, VkBuffer& uniformBuffer, VkDeviceMemory& uniformBufferMemory, void*& uniformBufferMapped);
	void CreateUniformBuffers(VkDeviceSize bufferSize, std::vector<VkBuffer>& uniformBuffers, std::vector<VkDeviceMemory>& uniformBuffersMemory, std::vector<void*>& uniformBuffersMapped);
	void CreateDescriptorPool(uint32_t nSets, VkDescriptorPool& descriptorPool);
	void CreateDescriptorSet(VkDescriptorSetLayout& descriptorSetLayout, VkDescriptorPool& descriptorPool, VkDescriptorSet& descriptorSet);
	void CreateDescriptorSets(VkDescriptorSetLayout& descriptorSetLayout, VkDescriptorPool& descriptorPool, std::vector<VkDescriptorSet>& descriptorSets);

	/* === Textures === */

	/* For creating texture(-like) images*/
	void CreateImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSample, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
	void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
	void TransitionImageLayout(VkCommandBuffer& commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
	void CopyBufferToImage(VkCommandBuffer& commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	void CopyImageToImage(VkCommandBuffer& commandBuffer, const ImVec2& extent, VkImage& srcImage, VkImage& dstImage);
	void CopyImageToImage(const ImVec2& extent, VkImage& srcImage, VkImage& dstImage);

	void CreateTextureImage(const Texture<uint8_t>& tex, uint32_t& mipLevels, VkImage& textureImage, VkDeviceMemory& textureImageMemory);
	void CreateTextureImageView(uint32_t mipLevels, VkImage& textureImage, VkImageView& textureImageView);
	void CreateTextureSampler(uint32_t mipLevels, VkSampler& textureSampler);

	void GenerateMipMaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
}