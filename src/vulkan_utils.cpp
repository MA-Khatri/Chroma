#include "vulkan_utils.h"

#include <stdlib.h>
#include <vector>
#include <fstream>

#include "stb_image.h"

#include "shader.h"


namespace VK
{
	/* === Namespace Globals === */
	extern VkInstance Instance = VK_NULL_HANDLE;
	extern VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
	extern VkDevice Device = VK_NULL_HANDLE;
	extern VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
	extern VkPipelineCache PipelineCache = VK_NULL_HANDLE;
	extern VkCommandPool TransferCommandPool = VK_NULL_HANDLE;
	extern VkCommandPool GraphicsCommandPool = VK_NULL_HANDLE;

	extern ImGui_ImplVulkanH_Window MainWindowData{};
	extern uint32_t MinImageCount = 2;
	extern uint32_t ImageCount = MinImageCount;
	extern bool SwapChainRebuild = false;

	extern uint32_t GraphicsQueueFamily = (uint32_t)-1;
	extern uint32_t ComputeQueueFamily = (uint32_t)-1;
	extern uint32_t TransferQueueFamily = (uint32_t)-1;

	extern VkQueue GraphicsQueue = VK_NULL_HANDLE;
	extern VkQueue ComputeQueue = VK_NULL_HANDLE;
	extern VkQueue TransferQueue = VK_NULL_HANDLE;

	extern VkDebugReportCallbackEXT DebugReport = VK_NULL_HANDLE;
	extern VkAllocationCallbacks* Allocator = nullptr;

	/* Per-frame-in-flight */
	extern std::vector<std::vector<VkCommandBuffer>> AllocatedGraphicsCommandBuffers{};
	extern std::vector<std::vector<std::function<void()>>> ResourceFreeQueue{};

	/*
	Unlike g_MainWindowData.FrameIndex, this is not the the swapchain image index
	and is always guaranteed to increase (eg. 0, 1, 2, 0, 1, 2)
	*/
	extern uint32_t CurrentFrameIndex = 0;


	/* ================================ */
	/* === Error Handling Utilities === */
	/* ================================ */
	
	void glfw_error_callback(int error, const char* description)
	{
		std::cerr << "GLFW Error " << error << " : " << description << std::endl;
	}

	
	void check_vk_result(VkResult err)
	{
		if (err == 0) return;
		std::cerr << "[Vulkan] Error: VkResult = " << err << std::endl;
		if (err < 0) exit(-1);
	}


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
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = GraphicsCommandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		VkResult err = vkAllocateCommandBuffers(Device, &allocInfo, &commandBuffer);
		check_vk_result(err);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		err = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		check_vk_result(err);

		return commandBuffer;
	}


	void FlushGraphicsCommandBuffer(VkCommandBuffer commandBuffer)
	{
		VkResult err = vkEndCommandBuffer(commandBuffer);
		check_vk_result(err);

		const uint64_t DEFAULT_FENCE_TIMEOUT = 100000000000;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		/* Create fence to ensure that the command buffer has finished executing */
		VkFence fence;
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = 0;
		err = vkCreateFence(Device, &fenceCreateInfo, nullptr, &fence);
		check_vk_result(err);

		err = vkQueueSubmit(GraphicsQueue, 1, &submitInfo, fence);
		check_vk_result(err);

		err = vkWaitForFences(Device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
		check_vk_result(err);

		vkDestroyFence(Device, fence, nullptr);
		vkFreeCommandBuffers(Device, GraphicsCommandPool, 1, &commandBuffer);
	}


	VkCommandBuffer GetTransferCommandBuffer()
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = TransferCommandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		VkResult err = vkAllocateCommandBuffers(Device, &allocInfo, &commandBuffer);
		check_vk_result(err);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		err = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		check_vk_result(err);

		return commandBuffer;
	}


	void FlushTransferCommandBuffer(VkCommandBuffer commandBuffer)
	{
		VkResult err = vkEndCommandBuffer(commandBuffer);
		check_vk_result(err);

		const uint64_t DEFAULT_FENCE_TIMEOUT = 100000000000;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		/* Create fence to ensure that the command buffer has finished executing */
		VkFence fence;
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = 0;
		err = vkCreateFence(Device, &fenceCreateInfo, nullptr, &fence);
		check_vk_result(err);

		err = vkQueueSubmit(TransferQueue, 1, &submitInfo, fence);
		check_vk_result(err);

		err = vkWaitForFences(Device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
		check_vk_result(err);

		vkDestroyFence(Device, fence, nullptr);
		vkFreeCommandBuffers(Device, TransferCommandPool, 1, &commandBuffer);
	}


	void SubmitResourceFree(std::function<void()>&& func)
	{
		ResourceFreeQueue[CurrentFrameIndex].emplace_back(func);
	}


	uint32_t GetVulkanMemoryType(VkMemoryPropertyFlags properties, uint32_t type_bits)
	{
		VkPhysicalDeviceMemoryProperties prop;
		vkGetPhysicalDeviceMemoryProperties(PhysicalDevice, &prop);
		for (uint32_t i = 0; i < prop.memoryTypeCount; i++)
		{
			if ((prop.memoryTypes[i].propertyFlags & properties) == properties && type_bits & (1 << i))
			{
				return i;
			}
		}

		return 0xffffffff;
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
		CreateTransientCommandPool(TransferQueueFamily, TransferCommandPool);
		CreateTransientCommandPool(GraphicsQueueFamily, GraphicsCommandPool);
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

		/* Update ImageCount in case its different... */
		ImageCount = wd->ImageCount;
	}


	bool IsExtensionAvailable(const ImVector<VkExtensionProperties>& properties, const char* extension)
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


	void CreateTransientCommandPool(uint32_t queueFamily, VkCommandPool& commandPool)
	{
		/* Create a transient command pool */
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		poolInfo.queueFamilyIndex = queueFamily;

		VkResult err = vkCreateCommandPool(Device, &poolInfo, nullptr, &commandPool);
		check_vk_result(err);
	}


	void CleanupVulkan()
	{
		vkDestroyCommandPool(Device, TransferCommandPool, Allocator);
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
	/* === ImGui Utility Functions === */
	/* =============================== */

	void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
	{
		VkResult err;

		VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
		VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
		err = vkAcquireNextImageKHR(Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			SwapChainRebuild = true;
			return;
		}
		check_vk_result(err);

		CurrentFrameIndex = (CurrentFrameIndex + 1) % MainWindowData.ImageCount;

		ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
		{
			err = vkWaitForFences(Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX); /* wait indefinitely instead of periodically checking */
			check_vk_result(err);

			err = vkResetFences(Device, 1, &fd->Fence);
			check_vk_result(err);
		}
		{
			/* Free resources in queue */
			for (auto& func : ResourceFreeQueue[CurrentFrameIndex])
			{
				func();
			}
			ResourceFreeQueue[CurrentFrameIndex].clear();
		}
		{
			/* Free command buffers allocated by Application::GetCommandBuffer */
			/* These use MainWindowData.FrameIndex and not CurrentFrameIndex because they're tied to the swapchain image index */
			auto& allocatedCommandBuffers = AllocatedGraphicsCommandBuffers[wd->FrameIndex];
			if (allocatedCommandBuffers.size() > 0)
			{
				vkFreeCommandBuffers(Device, fd->CommandPool, (uint32_t)allocatedCommandBuffers.size(), allocatedCommandBuffers.data());
				allocatedCommandBuffers.clear();
			}

			err = vkResetCommandPool(Device, fd->CommandPool, 0);
			check_vk_result(err);
			VkCommandBufferBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
			check_vk_result(err);
		}
		{
			VkRenderPassBeginInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			info.renderPass = wd->RenderPass;
			info.framebuffer = fd->Framebuffer;
			info.renderArea.extent.width = wd->Width;
			info.renderArea.extent.height = wd->Height;
			info.clearValueCount = 1;
			info.pClearValues = &wd->ClearValue;
			vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
		}

		/* Record dear imgui primitives into command buffer */
		ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

		/* Submit command buffer */
		vkCmdEndRenderPass(fd->CommandBuffer);
		{
			VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			VkSubmitInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.waitSemaphoreCount = 1;
			info.pWaitSemaphores = &image_acquired_semaphore;
			info.pWaitDstStageMask = &wait_stage;
			info.commandBufferCount = 1;
			info.pCommandBuffers = &fd->CommandBuffer;
			info.signalSemaphoreCount = 1;
			info.pSignalSemaphores = &render_complete_semaphore;

			err = vkEndCommandBuffer(fd->CommandBuffer);
			check_vk_result(err);
			err = vkQueueSubmit(GraphicsQueue, 1, &info, fd->Fence);
			check_vk_result(err);
		}
	}


	void FramePresent(ImGui_ImplVulkanH_Window* wd)
	{
		if (SwapChainRebuild)
		{
			return;
		}

		VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
		VkPresentInfoKHR info = {};
		info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		info.waitSemaphoreCount = 1;
		info.pWaitSemaphores = &render_complete_semaphore;
		info.swapchainCount = 1;
		info.pSwapchains = &wd->Swapchain;
		info.pImageIndices = &wd->FrameIndex;
		VkResult err = vkQueuePresentKHR(GraphicsQueue, &info);
		if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
		{
			SwapChainRebuild = true;
			return;
		}
		check_vk_result(err);
		wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount; /* Now we can use the next set of semaphores */
	}


	/* =============================== */
	/* === Layer Utility Functions === */
	/* =============================== */

	void CreateImage(ImVec2 extent, VkImage& image, VkDeviceMemory& memory)
	{
		VkResult err;

		/* Create the image */
		VkImageCreateInfo imageCreateInfo{};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = MainWindowData.SurfaceFormat.format;
		imageCreateInfo.extent.width = (uint32_t)extent.x;
		imageCreateInfo.extent.height = (uint32_t)extent.y;
		imageCreateInfo.extent.depth = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		err = vkCreateImage(Device, &imageCreateInfo, nullptr, &image);
		check_vk_result(err);

		/* Free the existing memory (if there is any) */
		vkFreeMemory(Device, memory, nullptr);

		/* Determine the new memory size and allocate */
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(Device, image, &memRequirements);

		VkMemoryAllocateInfo memAllocInfo{};
		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllocInfo.allocationSize = memRequirements.size;
		memAllocInfo.memoryTypeIndex = GetVulkanMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements.memoryTypeBits);
		err = vkAllocateMemory(Device, &memAllocInfo, nullptr, &memory);
		check_vk_result(err);

		/* Bind image data to the new memory allocation */
		err = vkBindImageMemory(Device, image, memory, 0);
		check_vk_result(err);
	}


	void CreateImages(uint32_t count, ImVec2 extent, std::vector<VkImage>& images, std::vector<VkDeviceMemory>& memory)
	{
		images.resize(count);
		memory.resize(count);

		for (uint32_t i = 0; i < count; i++)
		{
			CreateImage(extent, images[i], memory[i]);
		}
	}


	void CreateImageView(VkImage& image, VkImageView& imageView, VkFormat format)
	{
		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.image = image;
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = format;
		imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
		imageViewCreateInfo.subresourceRange.layerCount = 1;
		VkResult err = vkCreateImageView(Device, &imageViewCreateInfo, nullptr, &imageView);
		check_vk_result(err);
	}


	void CreateImageViews(std::vector<VkImage>& images, std::vector<VkImageView>& views)
	{
		views.resize(images.size());

		for (uint32_t i = 0; i < images.size(); i++)
		{
			CreateImageView(images[i], views[i], MainWindowData.SurfaceFormat.format);
		}
	}


	void CreateRenderPass(VkRenderPass& renderPass)
	{
		/* THIS FUNCTION SHOULD HAVE MORE OPTIONS TO SETUP THE RENDER PASS */

		VkResult err;

		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = MainWindowData.SurfaceFormat.format;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; /* we clear the color attachment with constants */
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0; /* index to attachment, e.g. "layout(location = 0) out vec4 outColor;" in frag shader code */
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		err = vkCreateRenderPass(Device, &renderPassInfo, nullptr, &renderPass);
		check_vk_result(err);
	}


	void CreateGraphicsPipeline(std::vector<std::string> shaderFiles , ImVec2 extent, VkRenderPass& renderPass, VkDescriptorSetLayout& descriptorSetLayout, VkPipelineLayout& layout, VkPipeline& pipeline)
	{
		VkResult err;

		/* ====== Shader Modules and Shader Stages ====== */
		//auto vertShader = ReadShaderFile(shaderFiles[0]);
		//auto fragShader = ReadShaderFile(shaderFiles[1]);
		//std::vector<VK::ShaderModule> shaderModules;
		//shaderModules.push_back({ CreateShaderModule(vertShader), VK_SHADER_STAGE_VERTEX_BIT });
		//shaderModules.push_back({ CreateShaderModule(fragShader), VK_SHADER_STAGE_FRAGMENT_BIT });
		//auto shaderStages = CreateShaderStages(shaderModules);

		auto shaderModules = CreateShaderModules(shaderFiles);
		auto shaderStages = CreateShaderStages(shaderModules);

		/* ====== Fixed Function Stages ====== */

		/* === Vertex Input === */
		/* This is where we state the bindings and attribute layout of input data */
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		/* === Input Assembly === */
		/* Where we define the type of primitive to draw (e.g. LINE_STRIP/TRIANGLE_LIST) */
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE; /* If true and using element (index) buffers, can use special index (e.g. 0xFFFF) to restart _STRIP topology */


		/* === Viewports and Scissors === */
		/* Viewport describes the region the frame will be rendered to. Scissor defines region where pixels are actually stored. */
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = extent.x;
		viewport.height = extent.y;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = { (uint32_t)extent.x, (uint32_t)extent.y };

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		/* === Rasterizer === */
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		/* === Multisampling === */
		/* We'll get back to this later? For now, disabled. */
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; /* optional */
		multisampling.pSampleMask = nullptr; /* optional */
		multisampling.alphaToCoverageEnable = VK_FALSE; /* optional */
		multisampling.alphaToOneEnable = VK_FALSE; /* optional */

		/* === Depth and stencil testing === */
		// TODO

		/* === Color Blending === */
		/*
		 * VkPipelineColorBlendAttachmentState = per attached framebuffer,
		 * VkPipelineColorBlendStateCreateInfo = *global* color blending settings
		 */
		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_AND;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; /* optional */
		colorBlending.blendConstants[1] = 0.0f; /* optional */
		colorBlending.blendConstants[2] = 0.0f; /* optional */
		colorBlending.blendConstants[3] = 0.0f; /* optional */

		/* === Dynamic States === */
		/* Since we are using them, these must be set before we draw! */
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();


		/* ====== Pipeline layout ====== */
		/* what we use to determine uniforms being sent to the shaders */
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0; /* optional */
		pipelineLayoutInfo.pPushConstantRanges = nullptr; /* optional */
		err = vkCreatePipelineLayout(Device, &pipelineLayoutInfo, nullptr, &layout);
		check_vk_result(err);


		/* ====== Pipeline creation ====== */
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = (uint32_t)shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; /* optional */
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = layout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0; /* index of the subpass where this graphics pipeline will be used */
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; /* optional -- used if you are creating derivative pipelines */
		pipelineInfo.basePipelineIndex = -1; /* optional */
		err = vkCreateGraphicsPipelines(Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
		check_vk_result(err);

		/* === Clean up === */
		DestroyShaderModules(shaderModules);
	}


	VkPipeline CreateGraphicsPipeline(std::vector<std::string> shaderFiles, ImVec2 extent, VkRenderPass& renderPass, VkDescriptorSetLayout& descriptorSetLayout, VkPipelineLayout& layout)
	{
		VkPipeline pipeline;
		CreateGraphicsPipeline(shaderFiles, extent, renderPass, descriptorSetLayout, layout, pipeline);
		return pipeline;
	}


	void CreateFrameBuffer(std::vector<VkImageView> attachments, VkRenderPass& renderPass, ImVec2 extent, VkFramebuffer& framebuffer)
	{
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = (uint32_t)attachments.size();
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = (uint32_t)extent.x;
		framebufferInfo.height = (uint32_t)extent.y;
		framebufferInfo.layers = 1;
		VkResult err = vkCreateFramebuffer(Device, &framebufferInfo, nullptr, &framebuffer);
		check_vk_result(err);
	}


	void CreateFrameBuffers(std::vector<VkImageView> attachments, VkRenderPass& renderPass, ImVec2 extent, uint32_t count, std::vector<VkFramebuffer>& framebuffers)
	{
		framebuffers.resize(count);

		for (uint32_t i = 0; i < count; i++)
		{
			CreateFrameBuffer(attachments, renderPass, extent, framebuffers[i]);
		}
	}


	void CreateViewportSampler(VkSampler* sampler)
	{
		VkSamplerCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		info.magFilter = VK_FILTER_LINEAR;
		info.minFilter = VK_FILTER_LINEAR;
		info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		info.minLod = -1000;
		info.maxLod = 1000;
		info.maxAnisotropy = 1.0f;
		VkResult err = vkCreateSampler(Device, &info, nullptr, sampler);
		check_vk_result(err);
	}


	/* =============== */
	/* === Buffers === */
	/* =============== */

	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(PhysicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		std::cerr << "Failed to find suitable memory type!" << std::endl;
		exit(-1);
	}


	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = GetTransferCommandBuffer();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0; /* optional */
		copyRegion.dstOffset = 0; /* optional */
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		FlushTransferCommandBuffer(commandBuffer);
	}


	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		/* Buffer creation */
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; /* might want to change this later if we want to do things like ray tracing using the same vertex data? (currently exclusive to graphics pipeline) */

		VkResult err = vkCreateBuffer(Device, &bufferInfo, nullptr, &buffer);
		check_vk_result(err);

		/* Memory allocation for buffer */
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(Device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		err = vkAllocateMemory(Device, &allocInfo, nullptr, &bufferMemory);
		check_vk_result(err);

		vkBindBufferMemory(Device, buffer, bufferMemory, 0);
	}


	void CreateVertexBuffer(const std::vector<Vertex> vertices, VkBuffer& vertexBuffer, VkDeviceMemory& vertexBufferMemory)
	{
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
	
		/* Create a staging buffer */
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		/* Transfer vertices to the staging buffer */
		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(Device, stagingBufferMemory);

		/* Create device local buffer and copy from staging buffer */
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
		CopyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		/* Cleanup the staging buffer that is no longer needed */
		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);
	}


	void CreateIndexBuffer(const std::vector<uint32_t> indices, VkBuffer& indexBuffer, VkDeviceMemory& indexBufferMemory)
	{
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		/* Create a staging buffer */
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		/* Transfer vertices to the staging buffer */
		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(Device, stagingBufferMemory);

		/* Create device local buffer and copy from staging buffer */
		CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
		CopyBuffer(stagingBuffer, indexBuffer, bufferSize);

		/* Cleanup the staging buffer that is no longer needed */
		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);
	}


	void CreateDescriptorSetLayout(std::vector<VkDescriptorSetLayoutBinding>& layoutBindings, VkDescriptorSetLayout& descriptorSetLayout)
	{
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
		layoutInfo.pBindings = layoutBindings.data();

		VkResult err = vkCreateDescriptorSetLayout(Device, &layoutInfo, nullptr, &descriptorSetLayout);
		check_vk_result(err);
	}


	void CreateUniformBuffers(VkDeviceSize bufferSize, std::vector<VkBuffer>& uniformBuffers, std::vector<VkDeviceMemory>& uniformBuffersMemory, std::vector<void*>& uniformBuffersMapped)
	{
		uniformBuffers.resize(ImageCount);
		uniformBuffersMemory.resize(ImageCount);
		uniformBuffersMapped.resize(ImageCount);

		for (size_t i = 0; i < ImageCount; i++)
		{
			CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
			vkMapMemory(Device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}


	void CreateDescriptorPool(VkDescriptorPool& descriptorPool)
	{
		std::vector<VkDescriptorPoolSize> poolSizes;

		VkDescriptorPoolSize imagesPoolSize{};
		imagesPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		imagesPoolSize.descriptorCount = static_cast<uint32_t>(ImageCount);
		poolSizes.push_back(imagesPoolSize);

		VkDescriptorPoolSize samplerPoolSize{};
		samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerPoolSize.descriptorCount = static_cast<uint32_t>(ImageCount);
		poolSizes.push_back(samplerPoolSize);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(ImageCount);

		VkResult err = vkCreateDescriptorPool(Device, &poolInfo, nullptr, &descriptorPool);
		check_vk_result(err);
	}


	void CreateDescriptorSets(VkDescriptorSetLayout& descriptorSetLayout, VkDescriptorPool& descriptorPool, std::vector<VkDescriptorSet>& descriptorSets)
	{
		std::vector<VkDescriptorSetLayout> layouts(ImageCount, descriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(ImageCount);
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(ImageCount);
		VkResult err = vkAllocateDescriptorSets(Device, &allocInfo, descriptorSets.data());
		check_vk_result(err);
	}


	/* ================ */
	/* === Textures === */
	/* ================ */

	void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0; /* optional -- for sparse images */

		VkResult err = vkCreateImage(Device, &imageInfo, nullptr, &image);
		check_vk_result(err);

		/* Allocate memory for the image and bind */
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(Device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);
		err = vkAllocateMemory(Device, &allocInfo, nullptr, &imageMemory);
		check_vk_result(err);

		vkBindImageMemory(Device, image, imageMemory, 0);
	}


	void CreateTextureImage(std::string filepath, VkImage& textureImage, VkDeviceMemory& textureImageMemory)
	{
		/* Load the image */
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(filepath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); /* load image with alpha channel even if it doesn't have one */
		VkDeviceSize imageSize = texWidth * texHeight * 4; /* 4 for RGBA channels */

		if (!pixels)
		{
			std::cerr << "[stb_image Error] Failed to load image " << filepath << " ! " << std::endl;
			exit(-1);
		}

		/* Create staging buffer */
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		/* Copy pixel data to the staging buffer */
		void* data;
		vkMapMemory(Device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(Device, stagingBufferMemory);

		/* Clean up original image from host side */
		stbi_image_free(pixels);

		/* Create the texture image */
		CreateImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		/* Copy the staging buffer to the texture image, adjusting the layouts as we go */
		TransitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		CopyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		TransitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		/* Clean up staging buffer */
		vkDestroyBuffer(Device, stagingBuffer, nullptr);
		vkFreeMemory(Device, stagingBufferMemory, nullptr);
	}


	void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
	{
		VkCommandBuffer commandBuffer = GetGraphicsCommandBuffer();

		/* Create a barrier */
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; /* Set these if you are transfering queue family ownership */
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		/* Determine access masks */
		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		/* Undefined -> transfer: transfer writes that don't need to wait on anything */
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		/* Transfer destination -> shader reading: shader reads should wait on transfer writes */
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			/* 
			 * May need to change the destination stage if we are using the texture earlier than the 
			 * fragment shader (e.g., using displacement maps in vertex/tes/geometry shader) 
			 */
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else
		{
			std::cerr << "TransitionImageLayout(): Unsupported layout transition! Old layout: " << oldLayout << ", new layout: " << newLayout << std::endl;
			exit(-1);
		}

		vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		FlushGraphicsCommandBuffer(commandBuffer);
	}


	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = GetTransferCommandBuffer();

		/* Specify which part of the buffer is going to be copied to which part of the image */
		VkBufferImageCopy region{};

		/* Offset into buffer at which values start */
		region.bufferOffset = 0;

		/* How pixels are laid out in memory -- 0, 0 indicates they are tightly packed */
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		/* I.e., which mip level do we want to copy from? */
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		/* What portion of the image do we want to copy? */
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		FlushTransferCommandBuffer(commandBuffer);
	}


	void CreateTextureImageView(VkImage& textureImage, VkImageView& textureImageView)
	{
		CreateImageView(textureImage, textureImageView, VK_FORMAT_R8G8B8A8_SRGB);
	}


	void CreateTextureSampler(VkSampler& textureSampler)
	{
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(PhysicalDevice, &properties);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy; /* Max available quality, worst performance */
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK; /* Only relevant if using a clamp to border addressing mode */
		samplerInfo.unnormalizedCoordinates = VK_FALSE; /* coordinates accessed in range [0, 1) */
		samplerInfo.compareEnable = VK_FALSE; /* used for things like percentage-closer filtering */
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		VkResult err = vkCreateSampler(Device, &samplerInfo, nullptr, &textureSampler);
		check_vk_result(err);
	}
}