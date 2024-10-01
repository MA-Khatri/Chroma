#pragma once

#include "optix_renderer.h"

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
	Camera m_Camera = Camera(100, 100, glm::vec3(0.0f, 10.0f, 5.0f), glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, 1.0), 45.0f);
	bool m_ViewportFocused = false;
	ImVec2 m_ViewportSize = ImVec2(10.0f, 10.0f);

	otx::Optix m_OptixRenderer = otx::Optix();
};