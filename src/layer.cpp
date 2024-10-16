#include "layer.h"
#include "stb_image_write.h"

#include <ctime>
#include <chrono>
#include <sstream>

#include "scene.h"
#include "math_helpers.h"


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


void Layer::SetupDebug(Application* app)
{
	/* Get a list of names for each drop down (aka combo) */
	for (auto& scene : app->GetScenes())
	{
		m_SceneNames.push_back(scene->m_SceneNames.at(scene->m_SceneType));
	}
	for (auto& controlMode : app->GetMainCamera()->m_ControlModeNames)
	{
		m_ControlModeNames.push_back(controlMode.second);
	}
	for (auto& projectionMode : app->GetMainCamera()->m_ProjectionModeNames)
	{
		m_ProjectionModeNames.push_back(projectionMode.second);
	}
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

	/* Make sure to reset the flag! */
	camera.m_CameraUIUpdate = false;

	ImGui::SeparatorText("Scene Settings");
	{
		/* Choose a scene */
		int selectedScene = app->GetSceneID();
		const char* selectedScenePreview = m_SceneNames[selectedScene].c_str();
		if (ImGui::BeginCombo("Scene", selectedScenePreview))
		{
			for (int n = 0; n < m_SceneNames.size(); n++)
			{
				const bool isSelected = (selectedScene == n);
				if (ImGui::Selectable(m_SceneNames[n].c_str(), isSelected)) selectedScene = n;

				if (isSelected) ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		if (selectedScene != app->GetSceneID()) camera.m_CameraUIUpdate = true;
		app->SetSceneID(selectedScene);
	}


	//if (ImGui::CollapsingHeader("Camera Settings"))
	ImGui::SeparatorText("Camera Settings");
	{
		ImGui::Checkbox("Link Cameras", &app->m_LinkCameras);
		
		/* Choose camera control mode */
		int selectedControlMode = camera.m_ControlMode;
		const char* controlModePreview = m_ControlModeNames[selectedControlMode].c_str();
		if (ImGui::BeginCombo("Control Mode", controlModePreview))
		{
			for (int n = 0; n < m_ControlModeNames.size(); n++)
			{
				const bool isSelected = (selectedControlMode == n);
				if (ImGui::Selectable(m_ControlModeNames[n].c_str(), isSelected)) selectedControlMode = n;

				if (isSelected) ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		if (selectedControlMode != camera.m_ControlMode) camera.m_CameraUIUpdate = true;
		camera.m_ControlMode = selectedControlMode;

		/* Choose camera projection mode */
		int selectedProjectionMode = camera.m_ProjectionMode;
		const char* projectionModePreview = m_ProjectionModeNames[selectedProjectionMode].c_str();
		if (ImGui::BeginCombo("Projection Mode", projectionModePreview))
		{
			for (int n = 0; n < m_ProjectionModeNames.size(); n++)
			{
				const bool isSelected = (selectedProjectionMode == n);
				if (ImGui::Selectable(m_ProjectionModeNames[n].c_str(), isSelected)) selectedProjectionMode = n;

				if (isSelected) ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		if (selectedProjectionMode != camera.m_ProjectionMode) camera.m_CameraUIUpdate = true;
		camera.m_ProjectionMode = selectedProjectionMode;

		/* Projection mode settings */
		if (camera.m_ProjectionMode == PROJECTION_MODE_PERSPECTIVE)
		{
			float fov = camera.m_VFoV;
			ImGui::DragFloat("Vertical FoV", &fov, 0.1f, camera.m_MinFoV, camera.m_MaxFoV);
			if (!Close(fov, camera.m_VFoV)) camera.m_CameraUIUpdate = true;
			camera.m_VFoV = fov;
		}
		else if (camera.m_ProjectionMode == PROJECTION_MODE_ORTHOGRAPHIC)
		{
			float scale = camera.m_OrthoScale;
			ImGui::DragFloat("Ortho Scale", &scale, camera.m_MinOrthoScale, camera.m_MinOrthoScale);
			if (!Close(scale, camera.m_OrthoScale)) camera.m_CameraUIUpdate = true;
			camera.m_OrthoScale = scale;
		}
		else if (camera.m_ProjectionMode == PROJECTION_MODE_THIN_LENS)
		{
			float fov = camera.m_VFoV;
			ImGui::DragFloat("Vertical FoV", &fov, 0.1f, camera.m_MinFoV, camera.m_MaxFoV);
			if (!Close(fov, camera.m_VFoV)) camera.m_CameraUIUpdate = true;
			camera.m_VFoV = fov;

			float defocusAngle = camera.m_DefocusAngle;
			ImGui::DragFloat("Defocus Angle", &defocusAngle, 0.01f, 0.0f, 10.0f);
			if (!Close(defocusAngle, camera.m_DefocusAngle)) camera.m_CameraUIUpdate = true;
			camera.m_DefocusAngle = defocusAngle;

			float focusDistance = camera.m_FocusDistance;
			ImGui::DragFloat("Focus Distance", &focusDistance, 0.1f, 0.1f, 100.0f);
			if (!Close(focusDistance, camera.m_FocusDistance)) camera.m_CameraUIUpdate = true;
			camera.m_FocusDistance = focusDistance;
		}

		/* Control mode settings */
		if (camera.m_ControlMode == CONTROL_MODE_FREE_FLY)
		{
			float posn[3] = { camera.m_Position.x, camera.m_Position.y, camera.m_Position.z };
			ImGui::DragFloat3("Camera Position", posn, 0.1f);
			glm::vec3 newPosn = glm::vec3(posn[0], posn[1], posn[2]);
			if (!Close(newPosn, camera.m_Position)) camera.m_CameraUIUpdate = true;
			camera.m_Position = newPosn;

			float ornt[3] = { camera.m_Orientation.x, camera.m_Orientation.y, camera.m_Orientation.z };
			ImGui::DragFloat3("Camera Orientation", ornt, 0.01f, -1.0f, 1.0f);
			glm::vec3 newOrnt = glm::normalize(glm::vec3(ornt[0], ornt[1], ornt[2]));
			if (!Close(newOrnt, camera.m_Orientation)) camera.m_CameraUIUpdate = true;
			camera.m_Orientation = newOrnt;
		}
		else if (camera.m_ControlMode == CONTROL_MODE_ORBIT)
		{
			float orig[3] = { camera.m_OrbitOrigin.x, camera.m_OrbitOrigin.y, camera.m_OrbitOrigin.z };
			ImGui::DragFloat3("Orbit Origin", orig, 0.1f);
			glm::vec3 newOrig = glm::vec3(orig[0], orig[1], orig[2]);
			if (!Close(newOrig, camera.m_OrbitOrigin)) camera.m_CameraUIUpdate = true;
			camera.m_OrbitOrigin = newOrig;

			float dist = camera.m_OrbitDistance;
			ImGui::DragFloat("Orbit Distance", &dist, 0.1f, 0.1f);
			if (!Close(dist, camera.m_OrbitDistance)) camera.m_CameraUIUpdate = true;
			camera.m_OrbitDistance = dist;

			float theta = camera.m_OrbitTheta;
			ImGui::DragFloat("Theta", &theta);
			if (!Close(theta, camera.m_OrbitTheta)) camera.m_CameraUIUpdate = true;
			camera.m_OrbitTheta = theta;

			float phi = camera.m_OrbitPhi;
			ImGui::DragFloat("Phi", &phi);
			if (!Close(phi, camera.m_OrbitPhi)) camera.m_CameraUIUpdate = true;
			camera.m_OrbitPhi = phi;
		}
	}	

	if (ImGui::Button("Take Screenshot"))
	{
		m_ScreenshotString = TakeScreenshot();
	}
	ImGui::Text(m_ScreenshotString.c_str());

}