#pragma once

#include "layer.h"

class RasterView : public Layer {

	virtual void OnUIRender() 
	{
		/* No padding on viewports */
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		{
			ImGui::Begin("Rasterized Viewport");
			{
				ImGui::BeginChild("Rasterized");
				{
					// TODO
				}
				ImGui::EndChild();
			}
			ImGui::End();

			ImGui::Begin("Ray Traced Viewport");
			{
				ImGui::BeginChild("Ray Traced");
				{
					// TODO
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

};
