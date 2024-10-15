#include "raytrace_view.h"
#include <algorithm>

RayTraceView::RayTraceView()
{ 
	// TODO?
}


RayTraceView::~RayTraceView()
{
	// TODO?
}


void RayTraceView::OnAttach(Application* app)
{
	SetupDebug(app);

	m_AppHandle = app;
	m_WindowHandle = app->GetWindowHandle();
	m_SceneID = app->GetSceneID();
	
	/* Create a separate Optix Renderer for each scene */
	for (auto& scene : m_AppHandle->GetScenes())
	{
		m_OptixRenderers.push_back(std::make_shared<otx::Optix>(scene));
	}
	m_OptixRenderer = m_OptixRenderers[m_SceneID];

	m_OptixRenderer->Resize(m_ViewportSize);
	m_RenderedImage.Resize(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y));
	m_RenderedImagePixels.resize(static_cast<size_t>(m_ViewportSize.x * m_ViewportSize.y));

	if (m_AppHandle->m_LinkCameras)	m_Camera = m_AppHandle->GetMainCamera();
	else m_Camera = m_LocalCamera;

	m_OptixRenderer->SetCamera(*m_Camera);
	m_OptixRenderer->SetSamplesPerRender(m_SamplesPerRender);
	m_OptixRenderer->SetMaxDepth(m_MaxDepth);
}

void RayTraceView::OnDetach()
{
	// TODO
}


void RayTraceView::OnUpdate()
{
	if (m_AppHandle->GetSceneID() != m_SceneID)
	{
		m_SceneID = m_AppHandle->GetSceneID();
		m_OptixRenderer = m_OptixRenderers[m_SceneID];
		m_OptixRenderer->Resize(m_ViewportSize);
		m_OptixRenderer->SetCamera(*m_Camera);
		m_OptixRenderer->SetSamplesPerRender(m_SamplesPerRender);
	}

	if (m_AppHandle->m_LinkCameras)	m_Camera = m_AppHandle->GetMainCamera();
	else m_Camera = m_LocalCamera;

	if (m_Camera->m_CameraUIUpdate)
	{
		m_Camera->UpdateViewMatrix();
		m_Camera->UpdateProjectionMatrix();
		if (m_Camera->m_ControlMode == Camera::ORBIT)
		{
			m_Camera->UpdateOrbit();
		}
		m_OptixRenderer->SetCamera(*m_Camera);
	}

	if (m_ViewportHovered)
	{
		bool updated = m_Camera->Inputs(m_WindowHandle);
		if (updated) m_OptixRenderer->SetCamera(*m_Camera);
	}

	if (m_ViewportFocused && m_AppHandle->m_FocusedWindow != Application::RayTracedViewport)
	{
		m_AppHandle->m_FocusedWindow = Application::RayTracedViewport;

		/* Set scroll callback for current camera */
		glfwSetWindowUserPointer(m_WindowHandle, m_Camera);
		glfwSetScrollCallback(m_WindowHandle, Camera::ScrollCallback);
		
		if (m_Camera->IsCameraDifferent(m_OptixRenderer->GetLastSetCamera()))
		{
			/* Reset camera for renderer if we switch back to ray trace view and camera settings have changed */
			m_OptixRenderer->SetCamera(*m_Camera);
		}
	}

	if (m_AppHandle->m_FocusedWindow == Application::RayTracedViewport)
	{
		m_OptixRenderer->Render();
		m_OptixRenderer->DownloadPixels(m_RenderedImagePixels.data()); /* Instead of downloading to host then re-uploading to GPU, can we upload directly? */
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
			m_ViewportFocused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

			ImGui::BeginChild("Ray Traced");
			{
				m_ViewportHovered = ImGui::IsWindowHovered();

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

			ImGui::SeparatorText("Ray Tracer");
			ImGui::Text("Total Accumulated Samples: %.1i", m_OptixRenderer->GetAccumulatedSampleCount());
		}
		ImGui::End();
	}
}


std::string RayTraceView::TakeScreenshot()
{
	int width = static_cast<int>(m_ViewportSize.x);
	int height = static_cast<int>(m_ViewportSize.y);

	std::vector<uint32_t> out = RotateAndFlip(m_RenderedImagePixels, width, height);

	/* Write to file */
	return WriteImageToFile(
		"output/" + GetDateTimeStr() + "_raytrace.png",
		width, height, 4,
		out.data(),
		width * 4
	);
}


void RayTraceView::OnResize(ImVec2 newSize)
{
	m_ViewportSize = newSize;

	ImVec2 mainWindowPos = ImGui::GetMainViewport()->Pos;
	ImVec2 viewportPos = ImGui::GetWindowPos();
	ImVec2 rPos = ImVec2(viewportPos.x - mainWindowPos.x, viewportPos.y - mainWindowPos.y);
	ImVec2 minR = ImGui::GetWindowContentRegionMin();
	ImVec2 maxR = ImGui::GetWindowContentRegionMax();
	m_Camera->m_ViewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
	m_Camera->m_ViewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
	m_Camera->UpdateProjectionMatrix(static_cast<int>(m_ViewportSize.x), static_cast<int>(m_ViewportSize.y));

	/* If cameras are linked, we still need to update the local camera */
	if (m_AppHandle->m_LinkCameras)
	{
		m_LocalCamera->m_ViewportContentMin = ImVec2(rPos.x + minR.x, rPos.y + minR.y);
		m_LocalCamera->m_ViewportContentMax = ImVec2(rPos.x + maxR.x, rPos.y + maxR.y);
		m_LocalCamera->UpdateProjectionMatrix(static_cast<int>(m_ViewportSize.x), static_cast<int>(m_ViewportSize.y));
	}

	m_OptixRenderer->Resize(m_ViewportSize);
	m_RenderedImage.Resize(static_cast<uint32_t>(m_ViewportSize.x), static_cast<uint32_t>(m_ViewportSize.y));
	m_RenderedImagePixels.resize(static_cast<size_t>(m_ViewportSize.x * m_ViewportSize.y));
}