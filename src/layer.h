#pragma once

#include <deque>
#include <memory>

#include <iostream>

#include "application.h"
#include "camera.h"

/* Forward decleration */
class Application;

/* 
 * A sliding buffer with m_MaxCount elements such that adding an element past 
 * m_MaxCount appends to the end and removes the first element
 */
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


/* Generate a vector of values from [start, stop) with the provided step size */
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


/* Get a string representing the current time in the format "YYYY-MM-DD_HH-MM-SS" */
std::string GetDateTimeStr();

/* Rotate 180 deg and flip the image that is represented as a vector of uint32s */
std::vector<uint32_t> RotateAndFlip(const std::vector<uint32_t>& in, uint32_t width, uint32_t height);

/* Use stb to write image data to provided filename. Returns saved image filepath or error message. */
std::string WriteImageToFile(std::string filename, int width, int height, int channels, const void* data, int stride);

class Layer
{
public:
	virtual ~Layer() = default;

	virtual void OnAttach(Application* app) {}
	virtual void OnDetach() {}

	virtual void OnUpdate() {}
	virtual void OnUIRender() {}

	virtual std::string TakeScreenshot() { return ""; }

protected:
	void CommonDebug(Application* app, ImVec2 viewport_size, const Camera& camera);

	int m_FrameStorageCount = 1001;
	SlidingBuffer<float> m_FrameTimes = SlidingBuffer<float>(m_FrameStorageCount);
	SlidingBuffer<float> m_FrameRates = SlidingBuffer<float>(m_FrameStorageCount);
	std::vector<float> m_FrameGraphX = arange<float>(0, (float)m_FrameStorageCount, 1);
	std::string m_ScreenshotString = "";
};