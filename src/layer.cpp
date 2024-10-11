#include "layer.h"
#include "stb_image_write.h"

#include <ctime>
#include <chrono>
#include <sstream>

std::string GetDateTimeStr()
{
	/* Get time */
	std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

	/* Format string */
	std::string s(30, '\0');
	std::strftime(&s[0], s.size(), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));

	/* Remove extra blank characters */
	std::string::iterator end_pos = std::remove(s.begin(), s.end(), '\0');
	s.erase(end_pos, s.end());

	return s;
}


std::vector<uint32_t> RotateAndFlip(const std::vector<uint32_t>& in, uint32_t width, uint32_t height)
{
	/* Rotate image ? */
	std::vector<uint32_t> copy = in;
	std::reverse(copy.begin(), copy.end());
	std::vector<uint32_t> out;
	out.resize(copy.size());

	/* Flip image horizontally */
	for (uint32_t j = 0; j < height; j++)
	{
		for (uint32_t i = 0; i < width; i++)
		{
			out[(j * width) + i] = copy[(j * width) + (width - i - 1)];
		}
	}

	return out;
}


std::string WriteImageToFile(std::string filename, int width, int height, int channels, const void* data, int stride)
{
	std::stringstream ss;
	if (stbi_write_png(filename.c_str(), width, height, channels, data, stride))
	{
		ss << "Saved image: " << filename << std::endl;
	}
	else
	{
		ss << "Error! Failed to save image: " << filename << std::endl;
	}
	return ss.str();
}


void Layer::CommonDebug(Application* app, ImVec2 viewport_size, const Camera& camera)
{
	ImGuiIO io = ImGui::GetIO();

	float frame_time = io.DeltaTime * 1000.0f;
	float frame_rate = 1.0f / io.DeltaTime;

	m_FrameTimes.Add(frame_time);
	m_FrameRates.Add(frame_rate);

	ImGui::Text("Frame Time: %.3f ms/frame (%.1f FPS)", frame_time, frame_rate);

	//ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 0));
	ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0, 0, 0, 0));
	ImPlot::SetNextAxisToFit(ImAxis_X1);
	ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 300);
	if (ImPlot::BeginPlot("##FrameRateGraph", ImVec2(-1, 150)))
	{
		//ImPlot::SetupAxes("", "FPS");
		//ImPlot::SetupAxisTicks(ImAxis_X1, 0, 1000, 11);
		ImPlot::PlotLine("##FrameRate", m_FrameGraphX.data(), m_FrameRates.GetItems().data(), m_FrameStorageCount);

		ImPlot::EndPlot();
	}

	ImGui::Text("Viewport Size :  %.1i x %.1i ", (int)viewport_size.x, (int)viewport_size.y);


	ImGui::Text("Camera Settings");
	ImGui::Checkbox("Link Cameras", &app->m_LinkCameras);
	ImGui::Text("\tVertical Field of View: %.1f deg", camera.vfov);
	ImGui::Text("\tCamera Position: X=%.3f, Y=%.3f, Z=%.3f", camera.position.x, camera.position.y, camera.position.z);
	ImGui::Text("\tCamera Orientation: X=%.3f, Y=%.3f, Z=%.3f", camera.orientation.x, camera.orientation.y, camera.orientation.z);

	if (ImGui::Button("Take Screenshot"))
	{
		m_ScreenshotString = TakeScreenshot();
	}
	ImGui::Text(m_ScreenshotString.c_str());

}