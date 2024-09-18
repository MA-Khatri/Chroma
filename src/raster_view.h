#pragma once

#include "layer.h"
#include "image.h"

#include "camera.h"

class RasterView : public Layer {

	virtual void OnAttach(GLFWwindow* window)
	{
		m_Window = window;
		m_Camera = new Camera(100, 100, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, 0.0, 1.0), 45.0f);

		m_Image = std::make_shared<Image>("./res/textures/teapot_normal.png");
	}

	virtual void OnDetach()
	{
		// TODO ?
	}

	virtual void OnUpdate()
	{
		if (m_ViewportFocused)
		{
			m_Camera->Inputs(m_Window);
		}
	}

	virtual void OnUIRender() 
	{
		ImVec2 viewport_size;

		/* No padding on viewports */
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		{
			ImGui::Begin("Rasterized Viewport");
			{
				ImGui::BeginChild("Rasterized");
				{
					m_ViewportFocused = ImGui::IsWindowFocused();
					viewport_size = ImGui::GetWindowSize();

					ImGui::Image(m_Image->GetDescriptorSet(), { (float)m_Image->GetWidth(), (float)m_Image->GetHeight()});
				}
				ImGui::EndChild();
			}
			ImGui::End();
		}
		/* Add back in padding for non-viewport ImGui */
		ImGui::PopStyleVar();

		ImGui::ShowDemoWindow();

		ImGui::Begin("Raster Debug Panel");
		{
			CommonDebug(viewport_size, m_Camera);
		}
		ImGui::End();
	}

private:
	GLFWwindow* m_Window;
	Camera* m_Camera;
	bool m_ViewportFocused = false;

	std::shared_ptr<Image> m_Image;
};
