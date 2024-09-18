#pragma once

#include <vector>
#include <memory>
#include <functional>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "implot.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "layer.h"

/* ========================= */
/* === Application Class === */
/* ========================= */

class Application
{
public:
	Application();
	~Application();

	static Application& Get(); /* Get application instance */

	void Run();
	void Close();

	void SetMenubarCallback(const std::function<void()>& menubarCallback) { m_MenubarCallback = menubarCallback; }
	void PushLayer(const std::shared_ptr<Layer>& layer) { m_LayerStack.emplace_back(layer); layer->OnAttach(); }

	GLFWwindow* GetWindowHandle() const { return m_WindowHandle; }
	static VkInstance GetInstance(); /* Get Vulkan instance */
	static VkPhysicalDevice GetPhysicalDevice();
	static VkDevice GetDevice();

private:
	void Init();
	void Shutdown();

	void RenderFrame();

private:
	GLFWwindow* m_WindowHandle;

	std::function<void()> m_MenubarCallback;
	std::vector<std::shared_ptr<Layer>> m_LayerStack;

	float m_TimeStep = 0.0f;
	float m_FrameTime = 0.0f;
	float m_LastFrameTime = 0.0f;

	bool m_Running = false;
};