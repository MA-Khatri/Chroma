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

#include "layer.h"


//#define APP_USE_UNLIMITED_FRAME_RATE
#ifdef _DEBUG
#define APP_USE_VULKAN_DEBUG_REPORT
#endif


/* ================================ */
/* === Error and Debug handlers === */
/* ================================ */

static void glfw_error_callback(int error, const char* description)
{
	std::cerr << "GLFW Error " << error << " : " << description << std::endl;
}


static void check_vk_result(VkResult err)
{
	if (err == 0)
	{
		return;
	}

	std::cerr << "[Vulkan] Error: VkResult = " << err << std::endl;

	if (err < 0)
	{
		abort();
	}
}


#ifdef APP_USE_VULKAN_DEBUG_REPORT
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(
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
	void PushLayer(const std::shared_ptr<Layer>& layer) { m_LayerStack.emplace_back(layer); layer->OnAttach(GetWindowHandle()); }

	GLFWwindow* GetWindowHandle() const { return m_WindowHandle; }
	static VkInstance GetInstance(); /* Get Vulkan instance */
	static VkPhysicalDevice GetPhysicalDevice();
	static VkDevice GetDevice();
	float GetTime();

	static VkCommandBuffer GetCommandBuffer(bool begin);
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