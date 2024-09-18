#pragma once

#include "layer.h"
#include "image.h"

class RayTraceView : public Layer 
{
	virtual void OnAttach()
	{
		// TODO ?
	}

	virtual void OnDetach()
	{
		// TODO ?
	}

	virtual void OnUpdate()
	{
		// TODO ?
	}

	virtual void OnUIRender()
	{
		ImVec2 viewport_size;

		/* No padding on viewports */
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		{
			ImGui::Begin("Ray Traced Viewport");
			{
				ImGui::BeginChild("Ray Traced");
				{
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
			CommonDebug(viewport_size);
		}
		ImGui::End();
	}
};