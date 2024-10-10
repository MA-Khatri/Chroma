#include "raytrace_view.h"


RayTraceView::RayTraceView(std::shared_ptr<Scene> scene)
	: m_OptixRenderer(otx::Optix(scene))
{ 
	// TODO?
}


RayTraceView::~RayTraceView()
{
	// TODO?
}


void RayTraceView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();

	m_OptixRenderer.Resize(m_ViewportSize);
	m_RenderedImage.Resize(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y));
	m_RenderedImagePixels.resize(static_cast<size_t>(m_ViewportSize.x * m_ViewportSize.y));

	if (m_AppHandle->m_LinkCameras)	m_Camera = m_AppHandle->GetMainCamera();
	else m_Camera = m_LocalCamera;

	m_OptixRenderer.SetCamera(*m_Camera);
	m_OptixRenderer.SetSamplesPerRender(1);
}

void RayTraceView::OnDetach()
{
	// TODO
}


void RayTraceView::OnUpdate()
{
	if (m_ViewportFocused) m_AppHandle->m_FocusedWindow = Application::RayTracedViewport;

	if (m_ViewportVisible)
	{
		if (m_AppHandle->m_LinkCameras)	m_Camera = m_AppHandle->GetMainCamera();
		else m_Camera = m_LocalCamera;

		if (m_ViewportHovered)
		{
			bool updated = m_Camera->Inputs(m_WindowHandle);
			if (updated) m_OptixRenderer.SetCamera(*m_Camera);
		}

		m_OptixRenderer.Render();
		m_OptixRenderer.DownloadPixels(m_RenderedImagePixels.data()); /* Instead of downloading to host then re-uploading to GPU, can we upload directly? */
		m_RenderedImage.SetData(m_RenderedImagePixels.data());
	}
}


void RayTraceView::OnUIRender()
{
	/* No padding on viewports */
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	{
		ImGui::Begin("Ray Traced Viewport");
		{
			ImGui::BeginChild("Ray Traced");
			{
				m_ViewportFocused = ImGui::IsWindowFocused();
				m_ViewportHovered = ImGui::IsWindowHovered();
				if (ImGui::IsWindowAppearing()) m_ViewportVisible = true;
				if (ImGui::IsWindowCollapsed()) m_ViewportVisible = false;

				ImVec2 newSize = ImGui::GetContentRegionAvail();
				if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y)
				{
					OnResize(newSize);
				}

				/* Note: we flip the image vertically to match Vulkan convention! */
				ImGui::Image(m_RenderedImage.GetDescriptorSet(), m_ViewportSize, ImVec2(0, 1), ImVec2(1, 0));
			}
			ImGui::EndChild();
		}
		ImGui::End();
	}
	/* Add back in padding for non-viewport ImGui */
	ImGui::PopStyleVar();

	if (m_AppHandle->m_FocusedWindow == Application::RayTracedViewport)
	{
		ImGui::Begin("Debug Panel");
		{
			CommonDebug(m_AppHandle, m_ViewportSize, *m_Camera);
		}
		ImGui::End();
	}
}


void RayTraceView::OnResize(ImVec2 newSize)
{
	m_ViewportSize = newSize;

	ImVec2 mainWindowPos = ImGui::GetMainViewport()->Pos;
	ImVec2 viewportPos = ImGui::GetWindowPos();
	ImVec2 rPos = ImVec2(viewportPos.x - mainWindowPos.x, viewportPos.y - mainWindowPos.y);
	ImVec2 minR = ImGui::GetWindowContentRegionMin();
	ImVec2 maxR = ImGui::GetWindowContentRegionMax();
	m_Camera->viewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
	m_Camera->viewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
	m_Camera->UpdateProjectionMatrix(static_cast<int>(m_ViewportSize.x), static_cast<int>(m_ViewportSize.y));

	/* If cameras are linked, we still need to update the local camera */
	if (m_AppHandle->m_LinkCameras)
	{
		m_LocalCamera->viewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
		m_LocalCamera->viewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
		m_LocalCamera->UpdateProjectionMatrix(static_cast<int>(m_ViewportSize.x), static_cast<int>(m_ViewportSize.y));
	}

	m_OptixRenderer.Resize(m_ViewportSize);
	m_RenderedImage.Resize(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y));
	m_RenderedImagePixels.resize(static_cast<size_t>(m_ViewportSize.x * m_ViewportSize.y));
}