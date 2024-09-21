#pragma once

#include <iostream>
#include <optional>
#include <functional>

#include "vulkan/vulkan.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"


#include "mesh.h"


//#define APP_USE_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define APP_USE_VULKAN_DEBUG_REPORT
#endif


namespace VK
{
	/* === Namespace Globals === */
	extern VkInstance Instance;
	extern VkPhysicalDevice PhysicalDevice;
	extern VkDevice Device;
	extern VkDescriptorPool DescriptorPool;
	extern VkPipelineCache PipelineCache;

	extern ImGui_ImplVulkanH_Window MainWindowData;
	extern uint32_t MinImageCount;
	extern bool SwapChainRebuild;

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
	void SubmitResourceFree(std::function<void()>&& func);

	uint32_t GetVulkanMemoryType(VkMemoryPropertyFlags properties, uint32_t type_bits);

	/* === Vulkan Setup Functions === */
	void SetupVulkan(ImVector<const char*> instance_extensions);
	void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height);

	void CreateInstance(ImVector<const char*> instance_extensions);
	void SelectPhysicalDevice();
	void GetQueueFamilies();
	void CreateLogicalDevice();
	void CreateDescriptorPool();

	void CleanupVulkan();
	void CleanupVulkanWindow();

	/* === ImGui Utility Functions === */
	void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data);
	void FramePresent(ImGui_ImplVulkanH_Window* wd);

	/* === Layer Utility Functions === */
	void CreateImage(ImVec2 extent, VkImage* image, VkDeviceMemory* memory);
	void CreateImages(uint32_t count, ImVec2 extent, std::vector<VkImage>* images, std::vector<VkDeviceMemory>* memory);

	void CreateImageView(VkImage* image, VkImageView* view);
	void CreateImageViews(uint32_t count, std::vector<VkImage>* images, std::vector<VkImageView>* views);

	void CreateRenderPass(VkRenderPass* renderPass);

	void CreateGraphicsPipeline(std::vector<std::string> shaderFiles, ImVec2 extent, VkRenderPass* renderPass, VkPipelineLayout* layout, VkPipeline* pipeline);

	void CreateFrameBuffer(std::vector<VkImageView> attachments, VkRenderPass* renderPass, ImVec2 extent, VkFramebuffer* framebuffer);
	void CreateFrameBuffers(std::vector<VkImageView> attachments, VkRenderPass* renderPass, ImVec2 extent, uint32_t count, std::vector<VkFramebuffer>* framebuffers);

	void CreateSampler(VkSampler* sampler);

	/* === Buffers === */
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	void CreateVertexBuffer(const std::vector<Vertex> vertices, VkBuffer* vertexBuffer, VkDeviceMemory* memory);

}