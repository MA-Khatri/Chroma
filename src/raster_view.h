#pragma once

#include "layer.h"
#include "image.h"

class RasterView : public Layer {

	virtual void OnAttach()
	{
		m_Image = std::make_shared<Image>("./res/textures/teapot_normal.png");
	}

	virtual void OnDetach()
	{
		// TODO ?
	}

	virtual void OnUpdate(float time_step)
	{
		// TODO ?
	}

	virtual void OnUIRender() 
	{
		/* No padding on viewports */
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		{
			ImGui::Begin("Rasterized Viewport");
			{
				ImGui::BeginChild("Rasterized");
				{
					ImGui::Image(m_Image->GetDescriptorSet(), { (float)m_Image->GetWidth(), (float)m_Image->GetHeight()});
				}
				ImGui::EndChild();
			}
			ImGui::End();
		}
		/* Add back in padding for non-viewport ImGui */
		ImGui::PopStyleVar();

		ImGui::ShowDemoWindow();

		ImGui::Begin("Debug Panel");
		{
			if (ImPlot::BeginPlot("My Plot"))
			{
				float x_data[] = { 0.0f, 1.0f, 2.0f, 3.0f };
				float y_data[] = { 1.0f, 2.0f, 0.5f, 1.5f };

				ImPlot::PlotLine("My Line", x_data, y_data, 4);
				ImPlot::EndPlot();
			}
		}
		ImGui::End();
	}

private:
	std::shared_ptr<Image> m_Image;
};
