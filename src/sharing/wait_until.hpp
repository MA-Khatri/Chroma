#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <stdexcept>
#include <thread>


#define MICROS(micros) std::chrono::microseconds(micros)

static bool wait_until(std::function<bool()> condition,
                       std::chrono::microseconds timeout = MICROS(10000),
                       std::chrono::microseconds check_interval = MICROS(100)) {
  auto start = std::chrono::microseconds(0);
  while (start < timeout) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(check_interval);
    start += check_interval;
  }
  return false;
}

template <typename T>
static std::optional<T> wait_until(std::function<T()> value, T expected_value,
                                   std::chrono::microseconds timeout = MICROS(10000),
                                   std::chrono::microseconds check_interval = MICROS(100)) {
  auto start = std::chrono::microseconds(0);
  while (start < timeout) {
    T result = value();
    if (result == expected_value) {
      return result;
    }
    std::this_thread::sleep_for(check_interval);
    start += check_interval;
  }
  return std::nullopt;
}
