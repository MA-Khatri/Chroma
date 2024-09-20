#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <stdlib.h> // abort
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "implot.h"
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>


#include "vulkan_utils.h"
#include "layer.h"



/* === Forward Declerations === */
class Layer;


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
	std::function<void()> GetMenubarCallback() { return m_MenubarCallback; }
	void PushLayer(const std::shared_ptr<Layer>& layer);

	GLFWwindow* GetWindowHandle() const { return m_WindowHandle; }
	float GetTime();

	static VkCommandBuffer GetCommandBuffer();
	static void FlushCommandBuffer(VkCommandBuffer commandBuffer);

	static void SubmitResourceFree(std::function<void()>&& func);

private:
	void Init();
	void Shutdown();
	void NextFrame();

private:
	GLFWwindow* m_WindowHandle;

	std::function<void()> m_MenubarCallback;
	std::vector<std::shared_ptr<Layer>> m_LayerStack;

	float m_TimeStep = 0.0f;
	float m_FrameTime = 0.0f;
	float m_LastFrameTime = 0.0f;

	bool m_Running = false;
};