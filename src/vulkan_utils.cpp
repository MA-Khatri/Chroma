#include "vulkan_utils.h"

#include <stdlib.h>
#include <iostream>
#include <vector>


namespace VK
{
	/* ================================ */
	/* === Error Handling Utilities === */
	/* ================================ */

	static void glfw_error_callback(int error, const char* description)
	{
		std::cerr << "GLFW Error " << error << " : " << description << std::endl;
	}


	static void check_vk_result(VkResult err)
	{
		if (err == 0) return;
		std::cerr << "[Vulkan] Error: VkResult = " << err << std::endl;
		if (err < 0) exit(-1);
	}

#ifdef APP_USE_VULKAN_DEBUG_REPORT
	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(
		VkDebugReportFlagsEXT flags,
		VkDebugReportObjectTypeEXT objectType,
		uint64_t object,
		size_t location,
		int32_t messageCode,
		const char* pLayerPrefix,
		const char* pMessage,
		void* pUserData
	) {
		(void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; /* Unused arguments */

		std::cerr << "[Vulkan] Debug report from ObjectType: " << objectType << " Message: " << pMessage << std::endl << std::endl;

		return VK_FALSE;
	}
#endif /* APP_USE_VULKAN_DEBUG_REPORT */


	/* ================================ */
	/* === Vulkan Utility Functions === */
	/* ================================ */

	VkCommandBuffer GetGraphicsCommandBuffer()
	{
		ImGui_ImplVulkanH_Window* wd = &MainWindowData;

		/* Use any command queue */
		VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;

		VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.commandPool = command_pool;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandBufferCount = 1;

		VkCommandBuffer& command_buffer = AllocatedGraphicsCommandBuffers[wd->FrameIndex].emplace_back();
		VkResult err = vkAllocateCommandBuffers(Device, &cmdBufAllocateInfo, &command_buffer);

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		err = vkBeginCommandBuffer(command_buffer, &begin_info);
		check_vk_result(err);

		return command_buffer;
	}


	void FlushGraphicsCommandBuffer(VkCommandBuffer commandBuffer)
	{
		const uint64_t DEFAULT_FENCE_TIMEOUT = 100000000000;

		VkSubmitInfo end_info = {};
		end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		end_info.commandBufferCount = 1;
		end_info.pCommandBuffers = &commandBuffer;
		auto err = vkEndCommandBuffer(commandBuffer);
		check_vk_result(err);

		/* Create fence to ensure that the command buffer has finished executing */
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = 0;
		VkFence fence;
		err = vkCreateFence(Device, &fenceCreateInfo, nullptr, &fence);
		check_vk_result(err);

		err = vkQueueSubmit(GraphicsQueue, 1, &end_info, fence);
		check_vk_result(err);

		err = vkWaitForFences(Device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
		check_vk_result(err);

		vkDestroyFence(Device, fence, nullptr);
	}


	void SubmitResourceFree(std::function<void()>&& func)
	{
		ResourceFreeQueue[CurrentFrameIndex].emplace_back(func);
	}


	/* ============================== */
	/* === Vulkan Setup Functions === */
	/* ============================== */
	
	void SetupVulkan(ImVector<const char*> instance_extensions)
	{
		CreateInstance(instance_extensions);
		SelectPhysicalDevice();
		GetQueueFamilies();
		CreateLogicalDevice();
		CreateDescriptorPool();
	}


	void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height)
	{
		wd->Surface = surface;

		/* Check for window system integration (WSI) support */
		VkBool32 res;
		vkGetPhysicalDeviceSurfaceSupportKHR(PhysicalDevice, GraphicsQueueFamily, wd->Surface, &res);
		if (res != VK_TRUE)
		{
			std::cerr << "Error no WSI support on selected physical device" << std::endl;
			exit(-1);
		}

		/* Select surface format */
		const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
		const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
		wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(PhysicalDevice, wd->Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

		/* Select present mode */
#ifdef APP_USE_UNLIMITED_FRAME_RATE
		VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
#else
		VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
#endif
		wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(PhysicalDevice, wd->Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));

		/* Create SwapChain, RenderPass, Framebuffer, etc. */
		IM_ASSERT(MinImageCount >= 2);
		ImGui_ImplVulkanH_CreateOrResizeWindow(Instance, PhysicalDevice, Device, wd, GraphicsQueueFamily, Allocator, width, height, MinImageCount);
	}


	static bool IsExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension)
	{
		for (const VkExtensionProperties& p : properties)
		{
			if (strcmp(p.extensionName, extension) == 0)
			{
				return true;
			}
		}
		return false;
	}


	void CreateInstance(ImVector<const char*> instance_extensions)
	{
		VkResult err;

		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

		/* Enumerate available extensions */
		uint32_t properties_count;
		ImVector<VkExtensionProperties> properties;
		vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
		properties.resize(properties_count);
		err = vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.Data);
		check_vk_result(err);

		/* Enable required extensions */
		if (IsExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
		{
			instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		}

		/* Enable validation layers (if in debug mode) */
#ifdef APP_USE_VULKAN_DEBUG_REPORT
		const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
		create_info.enabledLayerCount = 1;
		create_info.ppEnabledLayerNames = layers;
		instance_extensions.push_back("VK_EXT_debug_report");
#endif

		/* Create Vulkan instance */
		create_info.enabledExtensionCount = (uint32_t)instance_extensions.Size;
		create_info.ppEnabledExtensionNames = instance_extensions.Data;
		err = vkCreateInstance(&create_info, Allocator, &Instance);
		check_vk_result(err);

		/* Setup the debug report callback */
#ifdef APP_USE_VULKAN_DEBUG_REPORT
		auto f_vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(Instance, "vkCreateDebugReportCallbackEXT");
		IM_ASSERT(f_vkCreateDebugReportCallbackEXT != nullptr);
		VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
		debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		debug_report_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
		debug_report_ci.pfnCallback = debug_report;
		debug_report_ci.pUserData = nullptr;
		err = f_vkCreateDebugReportCallbackEXT(Instance, &debug_report_ci, Allocator, &DebugReport);
		check_vk_result(err);
#endif
	}


	void SelectPhysicalDevice()
	{
		uint32_t gpu_count;
		VkResult err = vkEnumeratePhysicalDevices(Instance, &gpu_count, nullptr);
		check_vk_result(err);
		IM_ASSERT(gpu_count > 0);

		ImVector<VkPhysicalDevice> gpus;
		gpus.resize(gpu_count);
		err = vkEnumeratePhysicalDevices(Instance, &gpu_count, gpus.Data);
		check_vk_result(err);

		/*
		If a number >1 of GPUs got reported, find discrete GPU if present, or use first one available.
		This covers most common cases (multi-gpu/integrated+dedicated graphics).
		*/
		for (VkPhysicalDevice& device : gpus)
		{
			VkPhysicalDeviceProperties properties;
			vkGetPhysicalDeviceProperties(device, &properties);

			if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				PhysicalDevice = device;
				return;
			}
		}

		/* Use first GPU (integrated) if a discrete one is unavailable */
		if (gpu_count > 0)
		{
			PhysicalDevice = gpus[0];
			return;
		}

		/* If we get here, no GPUs were found */
		std::cerr << "Failed to find GPU with Vulkan support!" << std::endl;
		PhysicalDevice = VK_NULL_HANDLE;
		return;
	}


	void GetQueueFamilies()
	{
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				GraphicsQueueFamily = i;
			}
			else if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				ComputeQueueFamily = i;
			}
			else if (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)
			{
				TransferQueueFamily = i;
			}
			i++;
		}

		/* If any of the queue families were not set, error */
		if (GraphicsQueueFamily == (uint32_t)-1 || ComputeQueueFamily == (uint32_t)-1 || TransferQueueFamily == (uint32_t)-1)
		{
			std::cerr << "Missing queue families on selected physical device!" << std::endl;
			exit(-1);
		}
	}


	void CreateLogicalDevice()
	{
		VkResult err;

		ImVector<const char*> device_extensions;
		device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		/* Enumerate physical device extension properties */
		uint32_t properties_count;
		ImVector<VkExtensionProperties> properties;
		vkEnumerateDeviceExtensionProperties(PhysicalDevice, nullptr, &properties_count, nullptr);
		properties.resize(properties_count);
		vkEnumerateDeviceExtensionProperties(PhysicalDevice, nullptr, &properties_count, properties.Data);

		/* Create one queue of each queue family with the same queue_priority */
		const float queue_priority[] = { 1.0f };
		VkDeviceQueueCreateInfo queue_info[3] = {}; 
		queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[0].queueFamilyIndex = GraphicsQueueFamily;
		queue_info[0].queueCount = 1;
		queue_info[0].pQueuePriorities = queue_priority;

		queue_info[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[1].queueFamilyIndex = ComputeQueueFamily;
		queue_info[1].queueCount = 1;
		queue_info[1].pQueuePriorities = queue_priority;

		queue_info[2].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[2].queueFamilyIndex = TransferQueueFamily;
		queue_info[2].queueCount = 1;
		queue_info[2].pQueuePriorities = queue_priority;

		/* Get *all* device features */
		VkPhysicalDeviceFeatures deviceFeatures{};
		vkGetPhysicalDeviceFeatures(PhysicalDevice, &deviceFeatures);

		/* Create the logical device */
		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
		create_info.pQueueCreateInfos = queue_info;
		create_info.enabledExtensionCount = (uint32_t)device_extensions.Size;
		create_info.ppEnabledExtensionNames = device_extensions.Data;
		create_info.pEnabledFeatures = &deviceFeatures; /* I.e., enable all device features */
		err = vkCreateDevice(PhysicalDevice, &create_info, Allocator, &Device);
		check_vk_result(err);

		/* Get queues */
		vkGetDeviceQueue(Device, GraphicsQueueFamily, 0, &GraphicsQueue);
		vkGetDeviceQueue(Device, ComputeQueueFamily, 0, &ComputeQueue);
		vkGetDeviceQueue(Device, TransferQueueFamily, 0, &TransferQueue);
	}


	void CreateDescriptorPool()
	{
		VkResult err;

		/*
		 * NOTE: for now, we're just making a bunch so we don't need to think about it. 
		 * Maybe better to make this dynamic in the future? Or somehow better allocation.
		 */

		VkDescriptorPoolSize pool_sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
		};
		VkDescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
		pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
		pool_info.pPoolSizes = pool_sizes;
		err = vkCreateDescriptorPool(Device, &pool_info, Allocator, &DescriptorPool);
		check_vk_result(err);
	}


	void CleanupVulkan()
	{
		vkDestroyDescriptorPool(Device, DescriptorPool, Allocator);

#ifdef APP_USE_VULKAN_DEBUG_REPORT
		/* Remove the debug report callback */
		auto f_vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(Instance, "vkDestroyDebugReportCallbackEXT");
		f_vkDestroyDebugReportCallbackEXT(Instance, DebugReport, Allocator);
#endif /* APP_USE_VULKAN_DEBUG_REPORT */

		vkDestroyDevice(Device, Allocator);
		vkDestroyInstance(Instance, Allocator);
	}


	void CleanupVulkanWindow()
	{
		ImGui_ImplVulkanH_DestroyWindow(Instance, Device, &MainWindowData, Allocator);
	}


	/* =============================== */
	/* === Layer Utility Functions === */
	/* =============================== */

}