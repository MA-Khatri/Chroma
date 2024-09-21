#include "vulkan_utils.h"

#include <stdlib.h>
#include <vector>
#include <fstream>

#include "shader.h"

namespace VK
{
	/* === Namespace Globals === */
	extern VkInstance Instance = VK_NULL_HANDLE;
	extern VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
	extern VkDevice Device = VK_NULL_HANDLE;
	extern VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
	extern VkPipelineCache PipelineCache = VK_NULL_HANDLE;

	extern ImGui_ImplVulkanH_Window MainWindowData{};
	extern uint32_t MinImageCount = 2;
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


	uint32_t GetVulkanMemoryType(VkMemoryPropertyFlags properties, uint32_t type_bits)
	{
		VkPhysicalDeviceMemoryProperties prop;
		vkGetPhysicalDeviceMemoryProperties(VK::PhysicalDevice, &prop);
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

	void CreateImage(ImVec2 extent, VkImage* image, VkDeviceMemory* memory)
	{
		VkResult err;

		/* Create the image */
		VkImageCreateInfo imageCreateInfo{};
		imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = VK::MainWindowData.SurfaceFormat.format;
		imageCreateInfo.extent.width = (uint32_t)extent.x;
		imageCreateInfo.extent.height = (uint32_t)extent.y;
		imageCreateInfo.extent.depth = 1;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.mipLevels = 1;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		err = vkCreateImage(Device, &imageCreateInfo, nullptr, image);
		VK::check_vk_result(err);

		/* Free the existing memory (if there is any) */
		vkFreeMemory(Device, *memory, nullptr);

		/* Determine the new memory size and allocate */
		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(Device, *image, &memRequirements);

		VkMemoryAllocateInfo memAllocInfo{};
		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllocInfo.allocationSize = memRequirements.size;
		memAllocInfo.memoryTypeIndex = VK::GetVulkanMemoryType(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements.memoryTypeBits);
		err = vkAllocateMemory(Device, &memAllocInfo, nullptr, memory);
		VK::check_vk_result(err);

		/* Bind image data to the new memory allocation */
		err = vkBindImageMemory(Device, *image, *memory, 0);
		VK::check_vk_result(err);
	}


	void CreateImages(uint32_t count, ImVec2 extent, std::vector<VkImage>* images, std::vector<VkDeviceMemory>* memory)
	{
		images->resize(count);
		memory->resize(count);

		for (uint32_t i = 0; i < count; i++)
		{
			CreateImage(extent, &(*images)[i], &(*memory)[i]);
		}
	}

	void CreateImageView(VkImage* image, VkImageView* view)
	{
		VkResult err;

		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.image = *image;
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = MainWindowData.SurfaceFormat.format;
		imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageViewCreateInfo.subresourceRange.levelCount = 1;
		imageViewCreateInfo.subresourceRange.layerCount = 1;
		err = vkCreateImageView(Device, &imageViewCreateInfo, nullptr, view);
		VK::check_vk_result(err);
	}


	void CreateImageViews(uint32_t count, std::vector<VkImage>* images, std::vector<VkImageView>* views)
	{
		views->resize(images->size());

		for (uint32_t i = 0; i < images->size(); i++)
		{
			CreateImageView(&(*images)[i], &(*views)[i]);
		}
	}


	void CreateRenderPass(VkRenderPass* renderPass)
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
		err = vkCreateRenderPass(Device, &renderPassInfo, nullptr, renderPass);
		VK::check_vk_result(err);
	}


	void CreateGraphicsPipeline(std::string vertexShaderFile, std::string fragmentShaderFile, ImVec2 extent, VkRenderPass* renderPass, VkPipelineLayout* layout, VkPipeline* pipeline)
	{
		VkResult err;

		/* ====== Shader Modules and Shader Stages ====== */
		//auto vertShaderCode = ReadShaderFile(vertexShaderFile);
		//auto fragShaderCode = ReadShaderFile(fragmentShaderFile);

		//VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		//VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

		VkShaderModule vertShaderModule = CreateShaderModule(vertexShaderFile);
		VkShaderModule fragShaderModule = CreateShaderModule(fragmentShaderFile);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main"; /* i.e., entry point */

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };


		/* ====== Fixed Function Stages ====== */

		/* === Vertex Input === */
		/* This is where we state the bindings and attribute layout of input data */
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr; /* optional */
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr; /* optional -- we'll set this later when drawing with actual vertex data */

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
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
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
		pipelineLayoutInfo.setLayoutCount = 0; /* optional */
		pipelineLayoutInfo.pSetLayouts = nullptr; /* optional */
		pipelineLayoutInfo.pushConstantRangeCount = 0; /* optional */
		pipelineLayoutInfo.pPushConstantRanges = nullptr; /* optional */
		err = vkCreatePipelineLayout(Device, &pipelineLayoutInfo, nullptr, layout);
		VK::check_vk_result(err);


		/* ====== Pipeline creation ====== */
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr; /* optional */
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = *layout;
		pipelineInfo.renderPass = *renderPass;
		pipelineInfo.subpass = 0; /* index of the subpass where this graphics pipeline will be used */
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; /* optional -- used if you are creating derivative pipelines */
		pipelineInfo.basePipelineIndex = -1; /* optional */
		err = vkCreateGraphicsPipelines(Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, pipeline);
		VK::check_vk_result(err);

		/* === Cleanup === */
		vkDestroyShaderModule(Device, vertShaderModule, nullptr);
		vkDestroyShaderModule(Device, fragShaderModule, nullptr);
	}


	void CreateFrameBuffer(std::vector<VkImageView> attachments, VkRenderPass* renderPass, ImVec2 extent, VkFramebuffer* framebuffer)
	{
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = *renderPass;
		framebufferInfo.attachmentCount = (uint32_t)attachments.size();
		framebufferInfo.pAttachments = attachments.data();
		framebufferInfo.width = (uint32_t)extent.x;
		framebufferInfo.height = (uint32_t)extent.y;
		framebufferInfo.layers = 1;
		VkResult err = vkCreateFramebuffer(Device, &framebufferInfo, nullptr, framebuffer);
		VK::check_vk_result(err);
	}


	void CreateFrameBuffers(std::vector<VkImageView> attachments, VkRenderPass* renderPass, ImVec2 extent, uint32_t count, std::vector<VkFramebuffer>* framebuffers)
	{
		framebuffers->resize(count);

		for (uint32_t i = 0; i < count; i++)
		{
			CreateFrameBuffer(attachments, renderPass, extent, &(*framebuffers)[i]);
		}
	}


	void CreateSampler(VkSampler* sampler)
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
		VK::check_vk_result(err);
	}
}