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
	void OnResize(ImVec2 newSize);

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera m_Camera;
	bool m_ViewportFocused = false;
	bool m_ViewportHovered = false;
	ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);


	otx::Optix m_OptixRenderer = otx::Optix(std::vector<Mesh>{LoadMesh("res/meshes/viking_room.obj")});
	Image m_RenderedImage = Image(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y), ImageFormat::RGBA, nullptr);
	std::vector<uint32_t> m_RenderedImagePixels;
};