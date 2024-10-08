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

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vulkan/vulkan_utils.h"
#include "layer.h"
#include "camera.h"


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
	Camera* GetMainCamera() const { return m_MainCamera; }
	float GetTime();

private:
	void Init();
	void Shutdown();
	void NextFrame();

public:
	enum {
		RasterizedViewport,
		RayTracedViewport
	};
	bool m_LinkCameras = true; /* Determines whether all viewports share the same camera (i.e., m_MainCamera) */
	int m_FocusedWindow = RasterizedViewport; /* Which window data should be used to display debug info? */

private:
	GLFWwindow* m_WindowHandle;

	std::function<void()> m_MenubarCallback;
	std::vector<std::shared_ptr<Layer>> m_LayerStack;

	float m_TimeStep = 0.0f;
	float m_FrameTime = 0.0f;
	float m_LastFrameTime = 0.0f;

	bool m_Running = false;

	/* The default, application-wide camera */
	Camera* m_MainCamera = new Camera();
};