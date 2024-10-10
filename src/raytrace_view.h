#pragma once

#include "optix/optix_renderer.h"

#include "vulkan/image.h"
#include "layer.h"
#include "camera.h"
#include "scene.h"

class RayTraceView : public Layer 
{
public:
	RayTraceView(std::shared_ptr<Scene> scene);
	~RayTraceView();

	/* Standard layer methods */
	virtual void OnAttach(Application* app);
	virtual void OnDetach();
	virtual void OnUpdate();
	virtual void OnUIRender();

	virtual void TakeScreenshot();

private:
	void OnResize(ImVec2 newSize);

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera = nullptr; /* Set on OnAttach() */
	Camera* m_LocalCamera = new Camera();
	bool m_ViewportFocused = false;
	bool m_ViewportHovered = false;
	ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

	otx::Optix m_OptixRenderer;
	Image m_RenderedImage = Image(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), ImageFormat::RGBA, nullptr);
	std::vector<uint32_t> m_RenderedImagePixels;
};