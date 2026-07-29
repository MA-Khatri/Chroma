#pragma once

#include <deque>
#include <filesystem>
#include <string>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_surface.h>
#include <SDL3_image/SDL_image.h>

#include <imgui.h>
#include <plog/Log.h>

#include "application.hpp"
#include "scene.hpp"

// Forward Declaration
class Application;

// A sliding buffer with m_MaxCount elements such that adding an element past
// m_MaxCount appends to the end and removes the first element
template <typename T> class SlidingBuffer {
public:
  SlidingBuffer(int maxCount) : m_MaxCount(maxCount), m_Deque(std::deque<T>()) {}

  void Add(T item) {
    if (m_Deque.size() == m_MaxCount) {
      m_Deque.pop_front();
    }
    m_Deque.push_back(item);
  }

  // Return vector of size m_MaxCount with empty leading elements = 0
  std::vector<T> GetItems() {
    std::vector<T> output(m_MaxCount);

    int cur_size = (int)m_Deque.size();
    int start = m_MaxCount - cur_size;

    for (int i = 0; i < m_MaxCount; ++i) {
      if (i < start) {
        output[i] = 0;
      } else {
        output[i] = m_Deque[i - start];
      }
    }

    return output;
  }

  // Return the last item in the deque
  T GetLastItem() {
    if (m_Deque.size() > 0) {
      return m_Deque[m_Deque.size() - 1];
    }

    return 0;
  }

private:
  int m_MaxCount;
  std::deque<T> m_Deque;
};

// Generate a vector of values from [start, stop) with the provided step size
template <typename T> std::vector<T> arange(T start, T stop, T step = 1) {
  std::vector<T> values;
  for (T value = start; value < stop; value += step) {
    values.push_back(value);
  }
  return values;
}

// Get a string representing the current time in the format
// "YYYY-MM-DD_HH-MM-SS"
std::string GetDateTimeStr();

// As the name suggests, used for flipping screenshots s.t. the origin is top
// left, not bottom left
template <typename T>
std::vector<T> FlipImageVertically(const std::vector<T> &in, int width, int height) {
  std::vector<T> out(in.size());

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      out[(j * width) + i] = in[(height - j - 1) * width + i];
    }
  }

  return out;
}

// Use SDL_image to write image data to provided filename. Returns a string with
// the generated saved image filepath or an error message so it can be displayed
// in ImGui.
template <typename T>
void WriteImageToFile(std::string filename, int width, int height, SDL_PixelFormat format,
                      std::vector<T> &pixelData) {
  PLOG_INFO << "Saving image to " << filename;

  // Check if the pixel data size is correct
  size_t bytesPerPixel = SDL_BYTESPERPIXEL(format);
  size_t pixelDataSize = pixelData.size() * sizeof(typename std::vector<T>::value_type);
  size_t providedSize = static_cast<size_t>(width * height) * bytesPerPixel;
  if (pixelDataSize != providedSize) {
    PLOG_ERROR << "Error! Pixel data size (" << pixelDataSize
               << ") does not match assumed size from given params: " << providedSize;
    return;
  }

  // Create an SDL surface from image data
  SDL_Surface *surface =
      SDL_CreateSurfaceFrom(width, height, format, static_cast<void *>(pixelData.data()),
                            /*pitch=*/width * bytesPerPixel);

  if (!surface) {
    PLOG_ERROR << "Error! SDL_CreateSurfaceFrom failed: " << SDL_GetError();
    return;
  }

  const std::string outputDir = std::string(ROOT_DIR) + "output/";
  try {
    // Creates the directory and any missing parent directories.
    // It automatically skips creation if the directory already exists.
    if (std::filesystem::create_directories(outputDir)) {
      PLOG_INFO << "Created output directory: " << outputDir;
    }
  } catch (const std::filesystem::filesystem_error &e) {
    PLOG_ERROR << "Error creating output directory " << outputDir << ": " << e.what();
  }

  // Save the surface to file as a PNG
  if (!IMG_SavePNG(surface, (outputDir + filename).c_str())) {
    PLOG_ERROR << "Error! IMG_SavePNG failed: " << SDL_GetError();
    return;
  }

  // Clean up the surface
  SDL_DestroySurface(surface);
}

class Layer {
public:
  virtual ~Layer() = default;

  virtual void OnAttach(Application *app) {}
  virtual void OnDetach() {}

  virtual void OnUpdate() {}
  virtual void OnUIRender() {}

  virtual void TakeScreenshot() {}

  void SetMouseWrapEnabled(bool enabled) { m_MouseWrapEnabled = enabled; }
  bool IsMouseWrapEnabled() const { return m_MouseWrapEnabled; }

protected:
  void CommonDebug(Application *app, std::shared_ptr<Camera> camera);

  void WrapMouseWithinRect(SDL_Window *window, const ImVec2 &rectMin, const ImVec2 &rectMax,
                           bool isActiveViewport, float edgeThreshold = 12.0f);

  // Frame rate/time graph buffers
  bool m_IncludeFrameRateGraph = false;
  int m_FrameGraphStorageCount = 1001;
  SlidingBuffer<float> m_FrameTimes = SlidingBuffer<float>(m_FrameGraphStorageCount);
  SlidingBuffer<float> m_FrameRates = SlidingBuffer<float>(m_FrameGraphStorageCount);
  std::vector<float> m_FrameGraphX = arange<float>(0, (float)m_FrameGraphStorageCount, 1);

  bool m_ViewportFocused = false;
  bool m_ViewportHovered = false;
  ImVec2 m_ViewportSize = ImVec2(400.0f, 400.0f);

  Application *m_AppHandle;
  SDL_Window *m_WindowHandle;

  std::string m_ViewportName;
  ImGuiID m_WindowID = 0;

  std::shared_ptr<Scene> m_CurrentScene;

private:
  bool m_MouseWrapEnabled = true;

  bool m_DragOriginatedInViewport = false;
  bool m_WasDragging = false;
};