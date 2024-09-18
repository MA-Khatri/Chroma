#pragma once

#include "application.h"

class Layer
{
public:
	virtual ~Layer() = default;

	virtual void OnAttach() {}
	virtual void OnDetach() {}

	virtual void OnUpdate() {}
	virtual void OnUIRender() {}

protected:
	void CommonDebug(ImVec2 viewport_size)
	{
		ImGuiIO io = ImGui::GetIO();

		ImGui::Text("Frame Time: %.3f ms/frame (%.1f FPS)", io.DeltaTime * 1000.0f, 1.0f / io.DeltaTime);
		ImGui::Text("Viewport Size :  %.1i x %.1i ", (int)viewport_size.x, (int)viewport_size.y);
	}
};