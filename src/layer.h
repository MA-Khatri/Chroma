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
	void CommonDebug(Application* app, ImVec2 viewport_size, Camera* camera);

	int frame_storage_count = 1001;
	SlidingBuffer<float> frame_times = SlidingBuffer<float>(frame_storage_count);
	SlidingBuffer<float> frame_rates = SlidingBuffer<float>(frame_storage_count);
	std::vector<float> x_axis = arange<float>(0, (float)frame_storage_count, 1);
};