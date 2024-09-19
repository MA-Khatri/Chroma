#pragma once

#include "layer.h"
#include "image.h"
#include "camera.h"

class RayTraceView : public Layer 
{
	virtual void OnAttach(Application* app);

	virtual void OnDetach();

	virtual void OnUpdate();

	virtual void OnUIRender();

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera;
	bool m_ViewportFocused = false;
	ImVec2 m_ViewportSize = ImVec2(10.0f, 10.0f);
};