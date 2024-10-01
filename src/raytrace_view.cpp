#include "raytrace_view.h"

void RayTraceView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();
}

void RayTraceView::OnDetach()
{
	// TODO ?
}


void RayTraceView::OnUpdate()
{
	if (m_ViewportFocused)
	{
		m_Camera.Inputs(m_WindowHandle);
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
				m_ViewportSize = ImGui::GetWindowSize();

				// TODO
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