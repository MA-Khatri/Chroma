#pragma once

#include <stdint.h>

struct SharedMemoryModelLayout {
public:
  // writing flag
  static constexpr size_t writing_flag_size = sizeof(uint32_t);
  // revision
  static constexpr size_t revision_number_size = sizeof(uint32_t);
  // tracking
  static constexpr size_t tracking_size = sizeof(uint32_t);
  // n_points
  static constexpr size_t n_points_size = sizeof(uint32_t);
  // pose
  static constexpr size_t pose_size = 16 * sizeof(float);
  // n_models
  static constexpr size_t n_models_size = sizeof(uint32_t);
  // positions
  static constexpr size_t single_position_size = 3 * sizeof(float);
  // normals
  static constexpr size_t single_normal_size = 3 * sizeof(float);
  // colors
  static constexpr size_t single_color_size = 3 * sizeof(float);
  // states
  static constexpr size_t single_state_size = sizeof(uint16_t);

  size_t Size() const {
    return writing_flag_size + revision_number_size + tracking_size + n_points_size + pose_size +
           n_models_size + n_models * n_models_size +
           n_points *
               (single_position_size + single_normal_size + single_color_size + single_state_size);
  }

  SharedMemoryModelLayout(uint32_t n_points, uint32_t n_models, uint8_t *data = nullptr)
      : n_points(n_points), n_models(n_models), data(data) {}

  void UpdateSize(int n_points, int n_models) {
    this->n_points = n_points;
    this->n_models = n_models;
  }

  uint8_t *writing_flag_ptr() { return data; }
  uint8_t *revision_ptr() { return writing_flag_ptr() + writing_flag_size; }
  uint8_t *tracking_ptr() { return revision_ptr() + revision_number_size; }
  uint8_t *n_points_ptr() { return tracking_ptr() + tracking_size; }
  uint8_t *pose_ptr() { return n_points_ptr() + n_points_size; }
  uint8_t *n_models_ptr() { return pose_ptr() + pose_size; }
  uint8_t *model_sizes_ptr() { return n_models_ptr() + n_models_size; }
  uint8_t *positions_ptr() { return model_sizes_ptr() + n_models_size * n_models; }
  uint8_t *normals_ptr() { return positions_ptr() + n_points * single_position_size; }
  uint8_t *colors_ptr() { return normals_ptr() + n_points * single_normal_size; }
  uint8_t *states_ptr() { return colors_ptr() + n_points * single_color_size; }

  static size_t LayoutSize(int n_points, int n_models) {
    SharedMemoryModelLayout layout(n_points, n_models);
    return layout.Size();
  }

private:
  uint8_t *data;
  uint32_t n_points;
  uint32_t n_models;
};
