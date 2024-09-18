#pragma once

#include "application.h"

#include <deque>


template <typename T> 
class SlidingBuffer
{
public:
	SlidingBuffer(int maxCount) : m_maxCount(maxCount), m_deque(std::deque<T>()) {}

	void Add(T item)
	{
		if (m_deque.size() == m_maxCount)
		{
			m_deque.pop_front();
		}
		m_deque.push_back(item);
	}

	/* Return vector of size m_maxCount with empty leading elements = 0 */
	std::vector<T> GetItems()
	{
		std::vector<T> output(m_maxCount);

		int cur_size = (int)m_deque.size();
		int start = m_maxCount - cur_size;

		for (int i = 0; i < m_maxCount; ++i)
		{
			if (i < start)
			{
				output[i] = 0;
			}
			else
			{
				output[i] = m_deque[i - start];
			}
		}

		return output;
	}

private:
	int m_maxCount;
	std::deque<T> m_deque;
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

	virtual void OnAttach() {}
	virtual void OnDetach() {}

	virtual void OnUpdate() {}
	virtual void OnUIRender() {}

protected:
	void CommonDebug(ImVec2 viewport_size)
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
	}

	int frame_storage_count = 1001;
	SlidingBuffer<float> frame_times = SlidingBuffer<float>(frame_storage_count);
	SlidingBuffer<float> frame_rates = SlidingBuffer<float>(frame_storage_count);
	std::vector<float> x_axis = arange<float>(0, frame_storage_count, 1);
};