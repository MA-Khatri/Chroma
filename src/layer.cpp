#include "layer.h"

void Layer::CommonDebug(Application* app, ImVec2 viewport_size, const Camera& camera)
{
	ImGuiIO io = ImGui::GetIO();

	float frame_time = io.DeltaTime * 1000.0f;
	float frame_rate = 1.0f / io.DeltaTime;

	frame_times.Add(frame_time);
	frame_rates.Add(frame_rate);

	ImGui::Text("Frame Time: %.3f ms/frame (%.1f FPS)", frame_time, frame_rate);

	//ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 0));
	ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0, 0, 0, 0));
	ImPlot::SetNextAxisToFit(ImAxis_X1);
	ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 300);
	if (ImPlot::BeginPlot("##FrameRateGraph", ImVec2(-1, 150)))
	{
		//ImPlot::SetupAxes("", "FPS");
		//ImPlot::SetupAxisTicks(ImAxis_X1, 0, 1000, 11);
		ImPlot::PlotLine("##FrameRate", x_axis.data(), frame_rates.GetItems().data(), frame_storage_count);

		ImPlot::EndPlot();
	}

	ImGui::Text("Viewport Size :  %.1i x %.1i ", (int)viewport_size.x, (int)viewport_size.y);


	ImGui::Text("Camera Settings");
	ImGui::Checkbox("Link Cameras", &app->m_LinkCameras);
	ImGui::Text("\tVertical Field of View: %.1f deg", camera.vfov);
	ImGui::Text("\tCamera Position: X=%.3f, Y=%.3f, Z=%.3f", camera.position.x, camera.position.y, camera.position.z);
	ImGui::Text("\tCamera Orientation: X=%.3f, Y=%.3f, Z=%.3f", camera.orientation.x, camera.orientation.y, camera.orientation.z);

}