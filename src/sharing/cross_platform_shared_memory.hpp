#pragma once

#include <cstdint>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#endif

// Inspired by https://github.com/itchio/shoom  (MIT license)

class CrossPlatformSharedMemory {
public:
  CrossPlatformSharedMemory();
  CrossPlatformSharedMemory(const CrossPlatformSharedMemory &) = delete;
  CrossPlatformSharedMemory &operator=(const CrossPlatformSharedMemory &) = delete;
  virtual ~CrossPlatformSharedMemory();

  bool Create(std::string name, int size);

  bool Open(std::string name, int size); // Normal use is read-only

  bool Close(); // When cleaning up

  bool IsOpen(); // For testing

  unsigned int Size(); // For testing

  uint8_t *Data(); // Mostly read-only access of data

  std::string Name(); // For testing

private:
  int _shm_fd;
  std::string _shm_name;
  int _shm_size;
  uint8_t *_shm_ptr;

  int _inject_error; // For unit testing purposes

#ifdef _WIN32
  HANDLE _handle;
#endif

  friend class TestCrossPlatformSharedMemory;
};
