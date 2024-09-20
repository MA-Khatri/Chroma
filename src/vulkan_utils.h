#pragma once

#include <iostream>
#include <optional>
#include <functional>

#include "vulkan/vulkan.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

//#define APP_USE_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define APP_USE_VULKAN_DEBUG_REPORT
#endif


namespace VK
{
	/* === Namespace Globals === */
	static VkInstance Instance = VK_NULL_HANDLE;
	static VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
	static VkDevice Device = VK_NULL_HANDLE;
	static VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
	static VkPipelineCache PipelineCache = VK_NULL_HANDLE;

	static ImGui_ImplVulkanH_Window MainWindowData;
	static uint32_t MinImageCount = 2;
	static bool SwapChainRebuild = false;

	static uint32_t GraphicsQueueFamily = (uint32_t)-1;
	static uint32_t ComputeQueueFamily = (uint32_t)-1;
	static uint32_t TransferQueueFamily = (uint32_t)-1;

	static VkQueue GraphicsQueue = VK_NULL_HANDLE;
	static VkQueue ComputeQueue = VK_NULL_HANDLE;
	static VkQueue TransferQueue = VK_NULL_HANDLE;

	static VkDebugReportCallbackEXT DebugReport = VK_NULL_HANDLE;
	static VkAllocationCallbacks* Allocator = nullptr;

	/* Per-frame-in-flight */
	static std::vector<std::vector<VkCommandBuffer>> AllocatedGraphicsCommandBuffers;
	static std::vector<std::vector<std::function<void()>>> ResourceFreeQueue;

	/*
	Unlike g_MainWindowData.FrameIndex, this is not the the swapchain image index
	and is always guaranteed to increase (eg. 0, 1, 2, 0, 1, 2)
	*/
	static uint32_t CurrentFrameIndex = 0;


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
	// TODO
}