#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

/* ========================= */
/* === Application Class === */
/* ========================= */

class Application
{
public:
	Application();
	~Application();

	void RenderFrame();

	GLFWwindow* GetWindowHandle() const { return m_WindowHandle; }

private:
	void Init();
	void Shutdown();

private:
	GLFWwindow* m_WindowHandle;
	ImGui_ImplVulkanH_Window* m_VulkanWindow;
	ImGuiIO m_ImGuiIO;

};