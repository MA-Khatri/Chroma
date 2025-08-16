#pragma once

#include <cstdint>

// Singleton
class Application {
public:
  static Application *GetInstance() {
    if (s_Instance == nullptr) {
      // Lazy initialize new instance
      s_Instance = new Application();
    }
    return s_Instance;
  }

  void Run();
  void Close();

  int64_t GetTimeNS(); // Current time in nanoseconds

private:
  static Application *s_Instance;
  Application();
  ~Application();

  // Remove copy constructor and assignment operator
  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;

  void Init();
  void Shutdown();
  void NextFrame();

  bool m_Running;

  // Time is stored in nanoseconds
  int64_t m_FrameTimeNS;
  int64_t m_TimeStepNS;
  int64_t m_LastFrameTimeNS;
};

constexpr uint64_t SecondsToNanoseconds(double seconds);
constexpr double NanosecondsToSeconds(uint64_t nanoseconds);