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

	virtual void OnUpdate(float time_step)
	{
		// TODO ?
	}

	virtual void OnUIRender()
	{
		/* No padding on viewports */
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		{
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
	}
};