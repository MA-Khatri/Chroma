#include "raytrace_view.h"

void RayTraceView::OnAttach(Application* app)
{
	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();

	m_Camera = new Camera(100, 100, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 1.0), 45.0f);
}

void RayTraceView::OnDetach()
{
	// TODO ?
}


void RayTraceView::OnUpdate()
{
	if (m_ViewportFocused)
	{
		m_Camera->Inputs(m_WindowHandle);
	}
}


void RayTraceView::OnUIRender()
{
	ImVec2 viewport_size;

	/* No padding on viewports */
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	{
		ImGui::Begin("Ray Traced Viewport");
		{
			ImGui::BeginChild("Ray Traced");
			{
				m_ViewportFocused = ImGui::IsWindowFocused();
				viewport_size = ImGui::GetWindowSize();

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
		CommonDebug(viewport_size, m_Camera);
	}
	ImGui::End();
}