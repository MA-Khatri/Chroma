#include "application.hpp"
#include "imgui.h"
#include "vulkan_engine/vulkan_utils.hpp"

#include <plog/Log.h>
#include <string>

constexpr uint64_t SecondsToNanoseconds(double seconds) { return seconds * SDL_NS_PER_SECOND; }

constexpr double NanosecondsToSeconds(uint64_t nanoseconds) {
  return nanoseconds / static_cast<double>(SDL_NS_PER_SECOND);
}

// ======================
// === Public Methods ===
// ======================

// Singleton instance
Application *Application::s_Instance = nullptr;

void Application::Run() {
  while (m_Controller->Running()) {
    NextFrame();
  }
}

void Application::PushLayer(const std::shared_ptr<Layer> &layer) {
  m_Layers.emplace_back(layer);
  layer->OnAttach(this);
}

void Application::PushScene(const std::shared_ptr<Scene> &scene) { m_Scenes.emplace_back(scene); }

void Application::SetActiveScene(int sceneIndex) {
  if (sceneIndex >= 0 && sceneIndex < static_cast<int>(m_Scenes.size())) {
    m_ActiveSceneIndex = sceneIndex;
  } else {
    PLOG_ERROR << "Invalid scene index: " << sceneIndex;
  }
}

std::shared_ptr<Scene> Application::GetActiveScene() { return m_Scenes[m_ActiveSceneIndex]; }

int64_t Application::GetTimeNS() {
  SDL_Time ns;
  if (SDL_GetCurrentTime(&ns))
    return ns;
  PLOG_ERROR << "Failed to get time in nanoseconds!" << SDL_GetError();
  return -1;
}

// =======================
// === Private Methods ===
// =======================

Application::Application() { Init(); }

Application::~Application() { Shutdown(); }

void Application::Init() {
  PLOG_DEBUG << "Initializing application...";

  // Setup SDL
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
    PLOG_FATAL << "Failed to initialize SDL: " << SDL_GetError();
    return;
  }

  // Create SDL window with Vulkan graphics context
  float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
  SDL_WindowFlags window_flags =
      SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
  m_WindowHandle =
      SDL_CreateWindow("Chroma", (int)(1280 * main_scale), (int)(720 * main_scale), window_flags);
  if (!m_WindowHandle) {
    PLOG_FATAL << "Failed to create SDL window: " << SDL_GetError();
    return;
  }

  ImVector<const char *> extensions;
  uint32_t sdl_extensions_count = 0;
  const char *const *sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extensions_count);
  for (uint32_t n = 0; n < sdl_extensions_count; n++)
    extensions.push_back(sdl_extensions[n]);
  vke::SetupVulkan(extensions);

  // Create window surface
  VkSurfaceKHR surface;
  VkResult err;
  if (!SDL_Vulkan_CreateSurface(m_WindowHandle, vke::Instance, vke::Allocator, &surface)) {
    PLOG_FATAL << "Failed to create Vulkan surface!";
    return;
  }

  // Create framebuffers
  int w, h;
  SDL_GetWindowSize(m_WindowHandle, &w, &h);
  ImGui_ImplVulkanH_Window *wd = &vke::MainWindowData;
  vke::SetupVulkanWindow(wd, surface, w, h);
  SDL_SetWindowPosition(m_WindowHandle, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
  SDL_ShowWindow(m_WindowHandle);

  vke::AllocatedGraphicsCommandBuffers.resize(wd->ImageCount);
  vke::ResourceFreeQueue.resize(wd->ImageCount);

  // Setup Dear Imgui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // Enables multiple windows for the program
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();

  // Setup scaling
  ImGuiStyle &style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }
  style.ScaleAllSizes(main_scale);
  style.FontScaleDpi = main_scale;

  // Setup Platform/Renderer backends
  ImGui_ImplSDL3_InitForVulkan(m_WindowHandle);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = vke::Instance;
  init_info.PhysicalDevice = vke::PhysicalDevice;
  init_info.Device = vke::Device;
  init_info.QueueFamily = vke::GraphicsQueueFamily;
  init_info.Queue = vke::GraphicsQueue;
  init_info.PipelineCache = vke::PipelineCache;
  init_info.DescriptorPool = vke::DescriptorPool;
  init_info.RenderPass = wd->RenderPass;
  init_info.Subpass = 0;
  init_info.MinImageCount = vke::MinImageCount;
  init_info.ImageCount = wd->ImageCount;
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  init_info.Allocator = vke::Allocator;
  init_info.CheckVkResultFn = vke::check_vk_result;
  ImGui_ImplVulkan_Init(&init_info);

  // Change default font (imgui fonts folder is copied to build dir)
  std::string font_path = "fonts/Roboto-Medium.ttf";
  float font_size = 16;
  ImFontConfig fontConfig;
  fontConfig.FontDataOwnedByAtlas = false;
  ImFont *default_font = io.Fonts->AddFontFromFileTTF(font_path.c_str(), font_size, &fontConfig);
  io.FontDefault = default_font;

  // Setup controller
  m_Controller = Controller::GetInstance();

  PLOG_DEBUG << "Application initialized successfully";
}

void Application::NextFrame() {
  ImGui_ImplVulkanH_Window *wd = &vke::MainWindowData;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  ImGuiIO &io = ImGui::GetIO();

  // Poll and handle SDL events (inputs, window resize, etc.)
  m_Controller->ProcessEvents();

  // Call the update functions for each layer
  for (auto &layer : m_Layers) {
    layer->OnUpdate();
  }

  // Resize swapchain if window(s) resized
  if (vke::SwapChainRebuild) {
    int w, h;
    SDL_GetWindowSize(m_WindowHandle, &w, &h);
    if (w > 0 && h > 0) {
      ImGui_ImplVulkan_SetMinImageCount(vke::MinImageCount);
      ImGui_ImplVulkanH_CreateOrResizeWindow(vke::Instance, vke::PhysicalDevice, vke::Device,
                                             &vke::MainWindowData, vke::GraphicsQueueFamily,
                                             vke::Allocator, w, h, vke::MinImageCount);
      vke::MainWindowData.FrameIndex = 0;

      // Clear allocated command buffers from here since entire pool is
      // destroyed
      vke::AllocatedGraphicsCommandBuffers.clear();
      vke::AllocatedGraphicsCommandBuffers.resize(vke::MainWindowData.ImageCount);

      vke::SwapChainRebuild = false;
    }
  }

  // Start the Dear ImGui frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();

  // Window contents
  {
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    // We are using the ImGuiWindowFlags_NoDocking flag to make the parent
    // window not dockable into, becuase it would be confusing to have two
    // docking targets within each other.
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
    auto menubar_callback = m_Controller->GetMenubarCallback();
    if (menubar_callback) {
      window_flags |= ImGuiWindowFlags_MenuBar;
    }

    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will
    // render our background and handle the pass-thru hole, so we ask Begin() to
    // not render a background.
    if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) {
      window_flags |= ImGuiWindowFlags_NoBackground;
    }

    // Important: note that we proceed even if Begin() returns false (aka window
    // is collapsed). This is because we want to keep our DockSpace() active. If
    // a DockSpace() is inactive, all active windows docked into it will lose
    // their parent and become undocked. We cannot preserve the docking
    // relationship between an active window and an inactive docking, otherwise
    // any change of dockspace/settings would lead to windows being stuck in
    // limbo and never being visible.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace", nullptr, window_flags);
    ImGui::PopStyleVar();

    ImGui::PopStyleVar(2);

    // Submit the DockSpace
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) {
      ImGuiID dockspace_id = ImGui::GetID("VulkanAppDockspace");
      ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    }

    if (menubar_callback) {
      if (ImGui::BeginMenuBar()) {
        menubar_callback();
        ImGui::EndMenuBar();
      }
    }

    // Call OnUIRender for each layer
    for (auto &layer : m_Layers) {
      layer->OnUIRender();
    }

    ImGui::End();
  }

  // Rendering
  ImGui::Render();
  ImDrawData *main_draw_data = ImGui::GetDrawData();
  const bool main_is_minimized =
      (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);
  wd->ClearValue.color.float32[0] = clear_color.x * clear_color.w;
  wd->ClearValue.color.float32[1] = clear_color.y * clear_color.w;
  wd->ClearValue.color.float32[2] = clear_color.z * clear_color.w;
  wd->ClearValue.color.float32[3] = clear_color.w;
  if (!main_is_minimized) {
    vke::FrameRender(wd, main_draw_data);
  }

  // Update and render additional platform windows
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
  }

  // Present main platform window
  if (!main_is_minimized) {
    vke::FramePresent(wd);
  }

  // Update timers
  int64_t timeNS = GetTimeNS();
  m_FrameTimeNS = timeNS - m_LastFrameTimeNS;
  constexpr int64_t MIN_TIMESTEP_FPS = SecondsToNanoseconds(1.0 / 30.0);
  m_TimeStepNS = std::min<int64_t>(m_FrameTimeNS, MIN_TIMESTEP_FPS);
  m_LastFrameTimeNS = timeNS;
}

void Application::Shutdown() {
  for (auto &layer : m_Layers) {
    layer->OnDetach();
  }

  VkResult err;
  err = vkDeviceWaitIdle(vke::Device);
  vke::check_vk_result(err);
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  // ImPlot::DestroyContext();
  ImGui::DestroyContext();

  vke::CleanupVulkanWindow();
  vke::CleanupVulkan();

  SDL_DestroyWindow(m_WindowHandle);
  SDL_Quit();
}