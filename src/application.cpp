#include "application.h"

/* ====================== */
/* === Vulkan Globals === */
/* ====================== */

static VkAllocationCallbacks*   g_Allocator = nullptr;
static VkInstance               g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice         g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice                 g_Device = VK_NULL_HANDLE;
static uint32_t                 g_GraphicsQueueFamily = (uint32_t)-1;
static VkQueue                  g_GraphicsQueue = VK_NULL_HANDLE;
static VkDebugReportCallbackEXT g_DebugReport = VK_NULL_HANDLE;
static VkPipelineCache          g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool         g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint32_t                 g_MinImageCount = 2;
static bool                     g_SwapChainRebuild = false;


/* Per-frame-in-flight */
static std::vector<std::vector<VkCommandBuffer>> s_AllocatedCommandBuffers;
static std::vector<std::vector<std::function<void()>>> s_ResourceFreeQueue;

/* 
Unlike g_MainWindowData.FrameIndex, this is not the the swapchain image index
and is always guaranteed to increase (eg. 0, 1, 2, 0, 1, 2)
*/
static uint32_t s_CurrentFrameIndex = 0;


/* Store a pointer to the application instance */
static Application* s_Instance = nullptr;


/* ==================== */
/* === Vulkan Utils === */
/* ==================== */

/* Checks if extension exists in the provided properties */
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


/* Returns the discrete GPU if one exists, else returns the first available GPU */
static VkPhysicalDevice SetupVulkan_SelectPhysicalDevice()
{
	uint32_t gpu_count;
	VkResult err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, nullptr);
	check_vk_result(err);
	IM_ASSERT(gpu_count > 0);

	ImVector<VkPhysicalDevice> gpus;
	gpus.resize(gpu_count);
	err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, gpus.Data);
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
			return device;
		}
	}

	/* Use first GPU (integrated) if a discrete one is unavailable */
	if (gpu_count > 0)
	{
		return gpus[0];
	}

	/* If we get here, no GPUs were found */
	std::cerr << "Failed to find GPU with Vulkan support!" << std::endl;
	return VK_NULL_HANDLE;
}


/* 
Creates a Vulkan instance with the provided extensions. 
Optionally adds validation layers debug callbacks if running in Debug mode. 
Also handles selecting a GPU, selecting the graphics queue, 
creating a logical device, and a descriptor pool.
*/
static void SetupVulkan(ImVector<const char*> instance_extensions)
{
	VkResult err;

	/* Create a Vulkan Instance */
	{
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
		err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
		check_vk_result(err);

		/* Setup the debug report callback */
#ifdef APP_USE_VULKAN_DEBUG_REPORT
		auto f_vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkCreateDebugReportCallbackEXT");
		IM_ASSERT(f_vkCreateDebugReportCallbackEXT != nullptr);
		VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
		debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		debug_report_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
		debug_report_ci.pfnCallback = debug_report;
		debug_report_ci.pUserData = nullptr;
		err = f_vkCreateDebugReportCallbackEXT(g_Instance, &debug_report_ci, g_Allocator, &g_DebugReport);
		check_vk_result(err);
#endif
	}

	/* Select physical device (GPU) */
	g_PhysicalDevice = SetupVulkan_SelectPhysicalDevice();

	/* Select graphics queue family */
	{
		uint32_t count;
		vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, nullptr);
		VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(sizeof(VkQueueFamilyProperties) * count);
		vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, queues);
		for (uint32_t i = 0; i < count; i++)
		{
			if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				g_GraphicsQueueFamily = i;
				break;
			}
			/* We can set the other queue families we want here later on... (make sure to remove break above) */
		}
		free(queues);
		IM_ASSERT(g_GraphicsQueueFamily != (uint32_t)-1);
	}

	/* Create Logical Device (with 1 queue) */
	{
		ImVector<const char*> device_extensions;
		device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

		/* Enumerate physical device extension */
		uint32_t properties_count;
		ImVector<VkExtensionProperties> properties;
		vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, nullptr);
		properties.resize(properties_count);
		vkEnumerateDeviceExtensionProperties(g_PhysicalDevice, nullptr, &properties_count, properties.Data);

		const float queue_priority[] = { 1.0f };
		VkDeviceQueueCreateInfo queue_info[1] = {}; /* We can add more queues here later */
		queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_info[0].queueFamilyIndex = g_GraphicsQueueFamily;
		queue_info[0].queueCount = 1;
		queue_info[0].pQueuePriorities = queue_priority;
		
		VkPhysicalDeviceFeatures deviceFeatures{};
		vkGetPhysicalDeviceFeatures(g_PhysicalDevice, &deviceFeatures);

		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
		create_info.pQueueCreateInfos = queue_info;
		create_info.enabledExtensionCount = (uint32_t)device_extensions.Size;
		create_info.ppEnabledExtensionNames = device_extensions.Data;
		create_info.pEnabledFeatures = &deviceFeatures; /* I.e., enable all device features */
		err = vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
		check_vk_result(err);
		vkGetDeviceQueue(g_Device, g_GraphicsQueueFamily, 0, &g_GraphicsQueue);
	}

	/* Create a descriptor pool */
	/* NOTE: for now, we're just making a bunch so we don't need to think about it. Maybe better to make this dynamic in the future. */
	{
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
		err = vkCreateDescriptorPool(g_Device, &pool_info, g_Allocator, &g_DescriptorPool);
		check_vk_result(err);
	}
}


/* 
Sets up Vulkan to render to the window surface. Also sets the presentation mode
and calls ImGui_ImplVulkanH_CreateOrResizeWindow() which handles creation of the 
swap chain, render pass, framebuffer, etc. 
*/
static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd, VkSurfaceKHR surface, int width, int height)
{
	wd->Surface = surface;

	/* Check for WSI support */
	VkBool32 res;
	vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_GraphicsQueueFamily, wd->Surface, &res);
	if (res != VK_TRUE)
	{
		std::cerr << "Error no WSI support on selected physical device" << std::endl;
		exit(-1);
	}

	/* Select Surface Format */
	const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
	const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
	wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(g_PhysicalDevice, wd->Surface, requestSurfaceImageFormat, (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), requestSurfaceColorSpace);

	/* Select Present Mode */
#ifdef APP_USE_UNLIMITED_FRAME_RATE
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR };
#else
	VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
#endif
	wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(g_PhysicalDevice, wd->Surface, &present_modes[0], IM_ARRAYSIZE(present_modes));

	/* Create SwapChain, RenderPass, Framebuffer, etc. */
	IM_ASSERT(g_MinImageCount >= 2);
	ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, wd, g_GraphicsQueueFamily, g_Allocator, width, height, g_MinImageCount);
}

/* Destroys descriptor pool, debug report callback, (logical) device, and Vulkan instance */
static void CleanupVulkan()
{
	vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

#ifdef APP_USE_VULKAN_DEBUG_REPORT
	/* Remove the debug report callback */
	auto f_vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(g_Instance, "vkDestroyDebugReportCallbackEXT");
	f_vkDestroyDebugReportCallbackEXT(g_Instance, g_DebugReport, g_Allocator);
#endif /* APP_USE_VULKAN_DEBUG_REPORT */

	vkDestroyDevice(g_Device, g_Allocator);
	vkDestroyInstance(g_Instance, g_Allocator);
}


/* Calls ImGui_ImplVulkanH_DestroyWindow with necessary arguments */
static void CleanupVulkanWindow()
{
	ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData, g_Allocator);
}


/* Renders the next ImGui frame */
static void FrameRender(ImGui_ImplVulkanH_Window* wd, ImDrawData* draw_data)
{
	VkResult err;

	VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
	VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
	err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
	if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
	{
		g_SwapChainRebuild = true;
		return;
	}
	check_vk_result(err);

	s_CurrentFrameIndex = (s_CurrentFrameIndex + 1) % g_MainWindowData.ImageCount;

	ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
	{
		err = vkWaitForFences(g_Device, 1, &fd->Fence, VK_TRUE, UINT64_MAX); /* wait indefinitely instead of periodically checking */
		check_vk_result(err);

		err = vkResetFences(g_Device, 1, &fd->Fence);
		check_vk_result(err);
	}
	{
		/* Free resources in queue */
		for (auto& func : s_ResourceFreeQueue[s_CurrentFrameIndex])
		{
			func();
		}
		s_ResourceFreeQueue[s_CurrentFrameIndex].clear();
	}
	{
		/* Free command buffers allocated by Application::GetCommandBuffer */
		/* These use g_MainWindowData.FrameIndex and not s_CurrentFrameIndex because they're tied to the swapchain image index */
		auto& allocatedCommandBuffers = s_AllocatedCommandBuffers[wd->FrameIndex];
		if (allocatedCommandBuffers.size() > 0)
		{
			vkFreeCommandBuffers(g_Device, fd->CommandPool, (uint32_t)allocatedCommandBuffers.size(), allocatedCommandBuffers.data());
			allocatedCommandBuffers.clear();
		}

		err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
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
		err = vkQueueSubmit(g_GraphicsQueue, 1, &info, fd->Fence);
		check_vk_result(err);
	}
}


/* Presents rendered ImGui frame with a call to vkQueuePresentKHR() */
static void FramePresent(ImGui_ImplVulkanH_Window* wd)
{
	if (g_SwapChainRebuild)
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
	VkResult err = vkQueuePresentKHR(g_GraphicsQueue, &info);
	if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR)
	{
		g_SwapChainRebuild = true;
		return;
	}
	check_vk_result(err);
	wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount; /* Now we can use the next set of semaphores */
}



/* ========================= */
/* === Application Class === */
/* ========================= */

Application::Application()
{
	Init();
}

Application::~Application()
{
	Shutdown();
}

void Application::PushLayer(const std::shared_ptr<Layer>& layer)
{
	m_LayerStack.emplace_back(layer); 
	layer->OnAttach(this);
}

Application& Application::Get()
{
	return *s_Instance;
}

ImGui_ImplVulkanH_Window* Application::GetMainWindowData()
{
	return &g_MainWindowData;
}

VkInstance Application::GetInstance()
{
	return g_Instance;
}

VkPhysicalDevice Application::GetPhysicalDevice()
{
	return g_PhysicalDevice;
}

VkDevice Application::GetDevice()
{
	return g_Device;
}

uint32_t Application::GetMinImageCount()
{
	return g_MinImageCount;
}

float Application::GetTime()
{
	return (float)glfwGetTime();
}

void Application::Init()
{
	/* Initialize GLFW */
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW!" << std::endl;
		abort();
	}

	/* Create window with Vulkan context */
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_WindowHandle = glfwCreateWindow(1280, 720, "Chroma", nullptr, nullptr);
	if (!glfwVulkanSupported())
	{
		std::cerr << "GLFW: Vulkan Not Supported" << std::endl;
		abort();
	}

	/* Determine required Vulkan extensions for GLFW and set up Vulkan w/ said extensions */
	ImVector<const char*> extensions;
	uint32_t extensions_count = 0;
	const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&extensions_count);
	for (uint32_t i = 0; i < extensions_count; i++)
	{
		extensions.push_back(glfw_extensions[i]);
	}
	SetupVulkan(extensions);

	/* Create Window Surface */
	VkSurfaceKHR surface;
	VkResult err = glfwCreateWindowSurface(g_Instance, m_WindowHandle, g_Allocator, &surface);
	check_vk_result(err);

	/* Create Framebuffers */
	int w, h;
	glfwGetFramebufferSize(m_WindowHandle, &w, &h);
	ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
	SetupVulkanWindow(wd, surface, w, h);


	s_AllocatedCommandBuffers.resize(wd->ImageCount);
	s_ResourceFreeQueue.resize(wd->ImageCount);

	/* Setup Dear ImGui context */
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImPlot::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       /* Enable Keyboard Controls */ 
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      /* Enable Gamepad Controls */
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           /* Enable Docking */
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         /* Enable Multi - Viewport / Platform Windows */
	//io.ConfigViewportsNoAutoMerge = true;
	//io.ConfigViewportsNoTaskBarIcon = true;

	/* Setup Dear ImGui style */
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsLight();

	/* When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones. */
	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	/* Setup Platform/Renderer backends */
	ImGui_ImplGlfw_InitForVulkan(m_WindowHandle, true);
	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = g_Instance;
	init_info.PhysicalDevice = g_PhysicalDevice;
	init_info.Device = g_Device;
	init_info.QueueFamily = g_GraphicsQueueFamily;
	init_info.Queue = g_GraphicsQueue;
	init_info.PipelineCache = g_PipelineCache;
	init_info.DescriptorPool = g_DescriptorPool;
	init_info.RenderPass = wd->RenderPass;
	init_info.Subpass = 0;
	init_info.MinImageCount = g_MinImageCount;
	init_info.ImageCount = wd->ImageCount;
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.Allocator = g_Allocator;
	init_info.CheckVkResultFn = check_vk_result;
	ImGui_ImplVulkan_Init(&init_info);

	/* Change default font */
	ImFontConfig fontConfig;
	fontConfig.FontDataOwnedByAtlas = false;
	float font_size = 16;
	ImFont* default_font = io.Fonts->AddFontFromFileTTF("external/imgui-docking/fonts/Roboto-Medium.ttf", font_size, &fontConfig);
	//ImFont* default_font = io.Fonts->AddFontFromFileTTF("external/imgui-docking/fonts/DroidSans.ttf", font_size, &fontConfig);
	//ImFont* default_font = io.Fonts->AddFontFromFileTTF("external/imgui-docking/fonts/Karla-Regular.ttf", font_size, &fontConfig);
	io.FontDefault = default_font;

	//ImGui_ImplVulkan_NewFrame();
}

void Application::Run()
{
	m_Running = true;

	while (!glfwWindowShouldClose(m_WindowHandle) && m_Running)
	{
		NextFrame();
	}
}

void Application::Close()
{
	m_Running = false;
}

void Application::NextFrame()
{
	ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	ImGuiIO& io = ImGui::GetIO();

	/* Poll and handle events (inputs, window resize, etc.) */
	glfwPollEvents();

	/* Call the update functions for each layer */
	for (auto& layer : m_LayerStack)
	{
		layer->OnUpdate();
	}

	/* Resize swapchain? */
	if (g_SwapChainRebuild)
	{
		int width, height;
		glfwGetFramebufferSize(m_WindowHandle, &width, &height);
		if (width > 0 && height > 0)
		{
			ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
			ImGui_ImplVulkanH_CreateOrResizeWindow(g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData, g_GraphicsQueueFamily, g_Allocator, width, height, g_MinImageCount);
			g_MainWindowData.FrameIndex = 0;

			// Clear allocated command buffers from here since entire pool is destroyed
			s_AllocatedCommandBuffers.clear();
			s_AllocatedCommandBuffers.resize(g_MainWindowData.ImageCount);

			g_SwapChainRebuild = false;
		}
	}

	/* Start the Dear ImGui frame */
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	/* Window contents */
	/* Adopted from the Cherno: https://github.com/StudioCherno/Walnut/blob/master/Walnut/src/Walnut/Application.cpp */
	{
		static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

		/* 
		We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
		because it would be confusing to have two docking targets within each others.
		*/
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
		if (m_MenubarCallback)
		{
			window_flags |= ImGuiWindowFlags_MenuBar;
		}

		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->WorkPos);
		ImGui::SetNextWindowSize(viewport->WorkSize);
		ImGui::SetNextWindowViewport(viewport->ID);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

		/*
		When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
		and handle the pass-thru hole, so we ask Begin() to not render a background.
		*/
		if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
		{
			window_flags |= ImGuiWindowFlags_NoBackground;
		}

		/*
		Important: note that we proceed even if Begin() returns false (aka window is collapsed).
		This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
		all active windows docked into it will lose their parent and become undocked.
		We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
		any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
		*/
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("DockSpace Demo", nullptr, window_flags);
		ImGui::PopStyleVar();

		ImGui::PopStyleVar(2);

		/* Submit the DockSpace */
		if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
		{
			ImGuiID dockspace_id = ImGui::GetID("VulkanAppDockspace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
		}

		if (m_MenubarCallback)
		{
			if (ImGui::BeginMenuBar())
			{
				m_MenubarCallback();
				ImGui::EndMenuBar();
			}
		}

		/* Call OnUIRender for each layer */
		for (auto& layer : m_LayerStack)
		{
			layer->OnUIRender();
		}

		ImGui::End();
	}


	/* Rendering */
	ImGui::Render();
	ImDrawData* main_draw_data = ImGui::GetDrawData();
	const bool main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);
	wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
	wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
	wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
	wd->ClearValue.color.float32[3] = clear_color.w;
	if (!main_is_minimized)
	{
		FrameRender(wd, main_draw_data);
	}

	/* Update and render additional platform windows */
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
	}

	/* Present main platform window */
	if (!main_is_minimized)
	{
		FramePresent(wd);
	}

	float time = GetTime();
	m_FrameTime = time - m_LastFrameTime;
	m_TimeStep = glm::min<float>(m_FrameTime, 0.0333f);
	m_LastFrameTime = time;
}


void Application::Shutdown()
{
	for (auto& layer : m_LayerStack)
	{
		layer->OnDetach();
	}

	VkResult err;

	err = vkDeviceWaitIdle(g_Device);
	check_vk_result(err);
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();

	CleanupVulkanWindow();
	CleanupVulkan();

	glfwDestroyWindow(m_WindowHandle);
	glfwTerminate();
}


VkCommandBuffer Application::GetCommandBuffer()
{
	ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;

	// Use any command queue
	VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;

	VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
	cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.commandPool = command_pool;
	cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = 1;

	VkCommandBuffer& command_buffer = s_AllocatedCommandBuffers[wd->FrameIndex].emplace_back();
	auto err = vkAllocateCommandBuffers(g_Device, &cmdBufAllocateInfo, &command_buffer);

	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	err = vkBeginCommandBuffer(command_buffer, &begin_info);
	check_vk_result(err);

	return command_buffer;
}


void Application::FlushCommandBuffer(VkCommandBuffer commandBuffer)
{
	const uint64_t DEFAULT_FENCE_TIMEOUT = 100000000000;

	VkSubmitInfo end_info = {};
	end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	end_info.commandBufferCount = 1;
	end_info.pCommandBuffers = &commandBuffer;
	auto err = vkEndCommandBuffer(commandBuffer);
	check_vk_result(err);

	// Create fence to ensure that the command buffer has finished executing
	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = 0;
	VkFence fence;
	err = vkCreateFence(g_Device, &fenceCreateInfo, nullptr, &fence);
	check_vk_result(err);

	err = vkQueueSubmit(g_GraphicsQueue, 1, &end_info, fence);
	check_vk_result(err);

	err = vkWaitForFences(g_Device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
	check_vk_result(err);

	vkDestroyFence(g_Device, fence, nullptr);
}


void Application::SubmitResourceFree(std::function<void()>&& func)
{
	s_ResourceFreeQueue[s_CurrentFrameIndex].emplace_back(func);
}