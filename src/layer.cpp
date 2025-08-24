#include "layer.hpp"

#include <chrono>
#include <ctime>

std::string GetDateTimeStr() {
  // Get current time as time_t
  auto now = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(now);

  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif

  // Format the time string
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", &tm);

  return std::string(buffer);
}