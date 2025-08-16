#include "application.hpp"

#include <SDL3/SDL_init.h>
#include <SDL3/SDL_video.h>
#include <cstdint>
#include <glm/glm.hpp>
#include <plog/Log.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include "vulkan/vulkan_utils.hpp"

constexpr uint64_t SecondsToNanoseconds(double seconds) {
  return seconds * SDL_NS_PER_SECOND;
}

constexpr double NanosecondsToSeconds(uint64_t nanoseconds) {
  return nanoseconds / static_cast<double>(SDL_NS_PER_SECOND);
}

///////////////////////////////////////////////////////////////////////////////
/// Public Methods
///////////////////////////////////////////////////////////////////////////////

// Singleton instance
Application *Application::s_Instance = nullptr;

void Application::Run() {
  m_Running = true;

  while (m_Running) {
    NextFrame();
    break;
  }
}

void Application::Close() { m_Running = false; }

int64_t Application::GetTimeNS() {
  SDL_Time ns;
  if (SDL_GetCurrentTime(&ns))
    return ns;
  PLOG_ERROR << "Failed to get time in nanoseconds!" << SDL_GetError();
  return -1;
}

///////////////////////////////////////////////////////////////////////////////
/// Private Methods
///////////////////////////////////////////////////////////////////////////////

Application::Application() { Init(); }

Application::~Application() { Shutdown(); }

void Application::Init() {

  // Setup SDL
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
    PLOG_FATAL << "Failed to initialize SDL: " << SDL_GetError();
    return;
  }

  // Create SDL window with Vulkan graphics context
  float main_scale = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
  SDL_WindowFlags window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                                 SDL_WINDOW_HIDDEN |
                                 SDL_WINDOW_HIGH_PIXEL_DENSITY;
  SDL_Window *window = SDL_CreateWindow("Chroma", (int)(1280 * main_scale),
                                        (int)(720 * main_scale), window_flags);
  if (window == nullptr) {
    PLOG_FATAL << "Failed to create SDL window: " << SDL_GetError();
    return;
  }

  ImVector<const char *> extensions;
  uint32_t sdl_extensions_count = 0;
  const char *const *sdl_extensions =
      SDL_Vulkan_GetInstanceExtensions(&sdl_extensions_count);
  for (uint32_t n = 0; n < sdl_extensions_count; n++)
    extensions.push_back(sdl_extensions[n]);
  // SetupVulkan(extensions); // TODO

  // Create window surface
  VkSurfaceKHR surface;
  VkResult err;
  if (!SDL_Vulkan_CreateSurface(window, 0, nullptr, nullptr)) {
    PLOG_FATAL << "Failed to create Vulkan surface!";
    return;
  }

  // Create framebuffers
  int w, h;
  SDL_GetWindowSize(window, &w, &h);
  ImGui_ImplVulkanH_Window *wd = &vk::MainWindowData;
  vk::SetupVulkanWindow(wd, surface, w, h);
  SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
  SDL_ShowWindow(window);

  // Setup Dear Imgui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableKeyboard; // Enable keyboard controls
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableGamepad; // Enable gamepad controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();

  // Setup scaling
  ImGuiStyle &style = ImGui::GetStyle();
  style.ScaleAllSizes(main_scale);
  style.FontScaleDpi = main_scale;

  // Setup Platform/Renderer backends
  ImGui_ImplSDL3_InitForVulkan(window);
  ImGui_ImplVulkan_InitInfo init_info = {};
  // init_info.Instance = g_Instance;
  // init_info.PhysicalDevice = g_PhysicalDevice;
  // init_info.Device = g_Device;
  // init_info.QueueFamily = g_QueueFamily;
  // init_info.Queue = g_Queue;
  // init_info.PipelineCache = g_PipelineCache;
  // init_info.DescriptorPool = g_DescriptorPool;
  // init_info.RenderPass = wd->RenderPass;
  // init_info.Subpass = 0;
  // init_info.MinImageCount = g_MinImageCount;
  // init_info.ImageCount = wd->ImageCount;
  // init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  // init_info.Allocator = g_Allocator;
  // init_info.CheckVkResultFn = check_vk_result;
  ImGui_ImplVulkan_Init(&init_info);
}

void Application::NextFrame() {
  // TODO

  int64_t timeNS = GetTimeNS();
  m_FrameTimeNS = timeNS - m_LastFrameTimeNS;

  constexpr int64_t MIN_TIMESTEP_FPS = SecondsToNanoseconds(1.0 / 30.0);

  m_TimeStepNS = glm::min<int64_t>(m_FrameTimeNS, MIN_TIMESTEP_FPS);
  m_LastFrameTimeNS = timeNS;
}

void Application::Shutdown() {
  // TODO
}