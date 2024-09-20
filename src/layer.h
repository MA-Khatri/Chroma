#pragma once

#include <deque>
#include <memory>

#include "application.h"
#include "camera.h"

/* Forward decleration */
class Application;


template <typename T> 
class SlidingBuffer
{
public:
	SlidingBuffer(int maxCount) : m_MaxCount(maxCount), m_Deque(std::deque<T>()) {}

	void Add(T item)
	{
		if (m_Deque.size() == m_MaxCount)
		{
			m_Deque.pop_front();
		}
		m_Deque.push_back(item);
	}

	/* Return vector of size m_MaxCount with empty leading elements = 0 */
	std::vector<T> GetItems()
	{
		std::vector<T> output(m_MaxCount);

		int cur_size = (int)m_Deque.size();
		int start = m_MaxCount - cur_size;

		for (int i = 0; i < m_MaxCount; ++i)
		{
			if (i < start)
			{
				output[i] = 0;
			}
			else
			{
				output[i] = m_Deque[i - start];
			}
		}

		return output;
	}

private:
	int m_MaxCount;
	std::deque<T> m_Deque;
};


template<typename T>
std::vector<T> arange(T start, T stop, T step = 1)
{
	std::vector<T> values;
	for (T value = start; value < stop; value += step)
	{
		values.push_back(value);
	}
	return values;
}


class Layer
{
public:
	virtual ~Layer() = default;

	virtual void OnAttach(Application* app) {}
	virtual void OnDetach() {}

	virtual void OnUpdate() {}
	virtual void OnUIRender() {}

protected:
	void CommonDebug(ImVec2 viewport_size, Camera* camera = nullptr)
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

		if (camera)
		{
			ImGui::Text("Camera Settings");
			ImGui::Text("\tVertical Field of View: %.1f deg", camera->vfov);
			ImGui::Text("\tCamera Position: X=%.3f, Y=%.3f, Z=%.3f", camera->position.x, camera->position.y, camera->position.z);
			ImGui::Text("\tCamera Orientation: X=%.3f, Y=%.3f, Z=%.3f", camera->orientation.x, camera->orientation.y, camera->orientation.z);
		}

	}

	int frame_storage_count = 1001;
	SlidingBuffer<float> frame_times = SlidingBuffer<float>(frame_storage_count);
	SlidingBuffer<float> frame_rates = SlidingBuffer<float>(frame_storage_count);
	std::vector<float> x_axis = arange<float>(0, (float)frame_storage_count, 1);
};