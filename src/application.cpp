#include "application.h"

/* Store a pointer to the application instance */
static Application* s_Instance = nullptr;


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

void Application::SetScenes(std::vector<std::shared_ptr<Scene>>& scenes)
{
	m_Scenes = scenes;
	m_SceneID = 0;
}

void Application::SetSceneID(int id)
{
	m_SceneID = id;
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
	vk::SetupVulkan(extensions);


	/* Create Window Surface */
	VkSurfaceKHR surface;
	VkResult err = glfwCreateWindowSurface(vk::Instance, m_WindowHandle, vk::Allocator, &surface);
	vk::check_vk_result(err);

	/* Create Framebuffers */
	int w, h;
	glfwGetFramebufferSize(m_WindowHandle, &w, &h);
	ImGui_ImplVulkanH_Window* wd = &vk::MainWindowData;
	vk::SetupVulkanWindow(wd, surface, w, h);


	vk::AllocatedGraphicsCommandBuffers.resize(wd->ImageCount);
	vk::ResourceFreeQueue.resize(wd->ImageCount);

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
	init_info.Instance = vk::Instance;
	init_info.PhysicalDevice = vk::PhysicalDevice;
	init_info.Device = vk::Device;
	init_info.QueueFamily = vk::GraphicsQueueFamily;
	init_info.Queue = vk::GraphicsQueue;
	init_info.PipelineCache = vk::PipelineCache;
	init_info.DescriptorPool = vk::DescriptorPool;
	init_info.RenderPass = wd->RenderPass;
	init_info.Subpass = 0;
	init_info.MinImageCount = vk::MinImageCount;
	init_info.ImageCount = wd->ImageCount;
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.Allocator = vk::Allocator;
	init_info.CheckVkResultFn = vk::check_vk_result;
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
	ImGui_ImplVulkanH_Window* wd = &vk::MainWindowData;
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
	if (vk::SwapChainRebuild)
	{
		int width, height;
		glfwGetFramebufferSize(m_WindowHandle, &width, &height);
		if (width > 0 && height > 0)
		{
			ImGui_ImplVulkan_SetMinImageCount(vk::MinImageCount);
			ImGui_ImplVulkanH_CreateOrResizeWindow(vk::Instance, vk::PhysicalDevice, vk::Device, &vk::MainWindowData, vk::GraphicsQueueFamily, vk::Allocator, width, height, vk::MinImageCount);
			vk::MainWindowData.FrameIndex = 0;

			/* Clear allocated command buffers from here since entire pool is destroyed */
			vk::AllocatedGraphicsCommandBuffers.clear();
			vk::AllocatedGraphicsCommandBuffers.resize(vk::MainWindowData.ImageCount);

			vk::SwapChainRebuild = false;
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
		vk::FrameRender(wd, main_draw_data);
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
		vk::FramePresent(wd);
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

	err = vkDeviceWaitIdle(vk::Device);
	vk::check_vk_result(err);
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();

	vk::CleanupVulkanWindow();
	vk::CleanupVulkan();

	glfwDestroyWindow(m_WindowHandle);
	glfwTerminate();
}