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


void Layer::CommonDebug(Application* app, ImVec2 viewport_size, Camera& camera)
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

	//if (ImGui::CollapsingHeader("Camera Settings"))
	ImGui::SeparatorText("Camera Settings");
	{
		ImGui::Checkbox("Link Cameras", &app->m_LinkCameras);
		
		camera.m_CameraUIUpdate = false;

		/* Choose camera control mode */
		const char* controlModes[] = { "Free fly", "Orbit" }; /* Make sure this matches order in Camera::ControlMode */
		static int selectedMode = camera.m_ControlMode;
		const char* preview = controlModes[selectedMode];
		if (ImGui::BeginCombo("Control Mode", preview))
		{
			for (int n = 0; n < IM_ARRAYSIZE(controlModes); n++)
			{
				const bool isSelected = (selectedMode == n);
				if (ImGui::Selectable(controlModes[n], isSelected)) selectedMode = n;

				if (isSelected) ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		if (selectedMode != camera.m_ControlMode) camera.m_CameraUIUpdate = true;
		camera.m_ControlMode = selectedMode;

		if (camera.m_ProjectionMode == Camera::PERSPECTIVE)
		{
			float fov = camera.m_VFoV;
			ImGui::DragFloat("Vertical FoV", &fov, 0.1f, camera.m_MinFoV, camera.m_MaxFoV);
			if (fov != camera.m_VFoV) camera.m_CameraUIUpdate = true;
			camera.m_VFoV = fov;
		}

		if (camera.m_ControlMode == Camera::FREE_FLY)
		{
			float posn[3] = { camera.m_Position.x, camera.m_Position.y, camera.m_Position.z };
			ImGui::DragFloat3("Camera Position", posn, 0.1f);
			glm::vec3 newPosn = glm::vec3(posn[0], posn[1], posn[2]);
			if (newPosn != camera.m_Position)camera.m_CameraUIUpdate = true;
			camera.m_Position = newPosn;

			float ornt[3] = { camera.m_Orientation.x, camera.m_Orientation.y, camera.m_Orientation.z };
			ImGui::DragFloat3("Camera Orientation", ornt, 0.01f, -1.0f, 1.0f);
			glm::vec3 newOrnt = glm::normalize(glm::vec3(ornt[0], ornt[1], ornt[2]));
			if (newOrnt != camera.m_Orientation) camera.m_CameraUIUpdate = true;
			camera.m_Orientation = newOrnt;
		}
		else if (camera.m_ControlMode == Camera::ORBIT)
		{
			float orig[3] = { camera.m_OrbitOrigin.x, camera.m_OrbitOrigin.y, camera.m_OrbitOrigin.z };
			ImGui::DragFloat3("Orbit Origin", orig, 0.1f);
			glm::vec3 newOrig = glm::vec3(orig[0], orig[1], orig[2]);
			if (newOrig != camera.m_OrbitOrigin)camera.m_CameraUIUpdate = true;
			camera.m_OrbitOrigin = newOrig;

			float dist = camera.m_OrbitDistance;
			ImGui::DragFloat("Orbit Distance", &dist, 0.1f, 0.1f);
			if (dist != camera.m_OrbitDistance) camera.m_CameraUIUpdate = true;
			camera.m_OrbitDistance = dist;

			float theta = camera.m_OrbitTheta;
			ImGui::DragFloat("Theta", &theta);
			if (theta != camera.m_OrbitTheta) camera.m_CameraUIUpdate = true;
			camera.m_OrbitTheta = theta;

			float phi = camera.m_OrbitPhi;
			ImGui::DragFloat("Phi", &phi);
			if (phi != camera.m_OrbitPhi) camera.m_CameraUIUpdate = true;
			camera.m_OrbitPhi = phi;
		}
	}	

	if (ImGui::Button("Take Screenshot"))
	{
		m_ScreenshotString = TakeScreenshot();
	}
	ImGui::Text(m_ScreenshotString.c_str());

}