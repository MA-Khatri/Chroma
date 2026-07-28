#pragma once

#include <stdint.h>

struct SharedMemoryLiveViewLayout {
public:
  // writing flag
  static constexpr size_t writing_flag_size = sizeof(uint32_t);
  // revision
  static constexpr size_t revision_number_size = sizeof(uint32_t);
  // width
  static constexpr size_t width_size = sizeof(uint32_t);
  // height
  static constexpr size_t height_size = sizeof(uint32_t);
  // pixel size
  static constexpr size_t single_pixel_size = sizeof(uint8_t) * 3;

  size_t Size() const {
    int num_pixels = width * height;
    return revision_number_size + writing_flag_size + width_size + height_size +
           num_pixels * single_pixel_size;
  }

  SharedMemoryLiveViewLayout(uint32_t width, uint32_t height, uint8_t *data = nullptr)
      : width(width), height(height), data(data) {}

  uint8_t *writing_flag_ptr() { return data; }
  uint8_t *revision_ptr() { return writing_flag_ptr() + writing_flag_size; }
  uint8_t *width_ptr() { return revision_ptr() + revision_number_size; }
  uint8_t *height_ptr() { return width_ptr() + width_size; }
  uint8_t *pixels_ptr() { return height_ptr() + height_size; }

  static size_t LayoutSize(uint32_t width, uint32_t height) {
    return SharedMemoryLiveViewLayout(width, height).Size();
  }

private:
  uint8_t *data;
  uint32_t width;
  uint32_t height;
};
