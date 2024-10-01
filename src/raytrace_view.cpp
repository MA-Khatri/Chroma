#include "raytrace_view.h"

void RayTraceView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();

	m_OptixRenderer.Resize(m_ViewportSize);
	m_RenderedImage.Resize(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y));
	m_RenderedImagePixels.resize(static_cast<size_t>(m_ViewportSize.x * m_ViewportSize.y));
}

void RayTraceView::OnDetach()
{
	// TODO
}


void RayTraceView::OnUpdate()
{
	if (m_ViewportHovered)
	{
		m_Camera.Inputs(m_WindowHandle);
	}

	m_OptixRenderer.Render();
	m_OptixRenderer.DownloadPixels(m_RenderedImagePixels.data());
	m_RenderedImage.SetData(m_RenderedImagePixels.data());
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

				ImVec2 newSize = ImGui::GetContentRegionAvail();
				if (m_ViewportSize.x != newSize.x || m_ViewportSize.y != newSize.y)
				{
					OnResize(newSize);
				}

				ImGui::Image(m_RenderedImage.GetDescriptorSet(), m_ViewportSize);
			}
			ImGui::EndChild();
		}
		ImGui::End();
	}
	/* Add back in padding for non-viewport ImGui */
	ImGui::PopStyleVar();

	ImGui::Begin("Ray Trace Debug Panel");
	{
		CommonDebug(m_ViewportSize, m_Camera);
	}
	ImGui::End();
}


void RayTraceView::OnResize(ImVec2 newSize)
{
	ImVec2 mainWindowPos = ImGui::GetMainViewport()->Pos;
	ImVec2 viewportPos = ImGui::GetWindowPos();
	ImVec2 rPos = ImVec2(viewportPos.x - mainWindowPos.x, viewportPos.y - mainWindowPos.y);
	ImVec2 minR = ImGui::GetWindowContentRegionMin();
	ImVec2 maxR = ImGui::GetWindowContentRegionMax();
	m_Camera.viewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
	m_Camera.viewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);

	m_ViewportSize = newSize;

	m_OptixRenderer.Resize(m_ViewportSize);
	m_RenderedImage.Resize(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y));
	m_RenderedImagePixels.resize(static_cast<size_t>(m_ViewportSize.x * m_ViewportSize.y));
}