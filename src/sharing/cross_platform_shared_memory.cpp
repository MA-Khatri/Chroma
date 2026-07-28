#include "cross_platform_shared_memory.hpp"

#include <plog/Log.h>

#ifndef _WIN32
#include <fcntl.h>    // for O_* constants
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // for mode constants
#include <unistd.h>   // unlink

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#endif

#ifdef _WIN32

// Shared memory is based on memory mapped file
CrossPlatformSharedMemory::CrossPlatformSharedMemory() {
  _shm_ptr = nullptr;
  _shm_size = 0;
  _shm_fd = -1;
  _handle = nullptr;

  _inject_error = 0; // No errors injected
}

CrossPlatformSharedMemory::~CrossPlatformSharedMemory() { Close(); }

bool CrossPlatformSharedMemory::Create(std::string name, int size) {
  _shm_name = name;
  _shm_size = size;

  DWORD size_high_order = 0;
  DWORD size_low_order = static_cast<DWORD>(_shm_size);
  _handle = CreateFileMappingA(INVALID_HANDLE_VALUE, // use paging file
                               NULL,                 // default security
                               PAGE_READWRITE,       // read/write access
                               size_high_order, size_low_order,
                               _shm_name.c_str() // name of mapping object
  );

  if (!_handle || _inject_error == -1) {
    PLOG_ERROR << "Creation of shm failed: name=" << _shm_name << " size=" << _shm_size;
    return false;
  }

  DWORD access = FILE_MAP_ALL_ACCESS;

  _shm_ptr = static_cast<uint8_t *>(MapViewOfFile(_handle, access, 0, 0, _shm_size));

  if (!_shm_ptr || _inject_error == -2) {
    PLOG_ERROR << "Mapping of shm failed: name=" << name << " size=" << size;
    return false;
  }

  PLOG_INFO << "Created: " << _shm_name << "  size: " << _shm_size;
  return true;
}

// Read-write memory access
bool CrossPlatformSharedMemory::Open(std::string name, int size) {
  _shm_name = name;
  _shm_size = size;

  DWORD access = FILE_MAP_ALL_ACCESS;
  _handle = OpenFileMappingA(access,           // read/write access
                             FALSE,            // do not inherit the name
                             _shm_name.c_str() // name of mapping object
  );

  if (!_handle || (_inject_error == -3)) {
    PLOG_ERROR << "Open shm failed: name=" << _shm_name;
    return false;
  }

  _shm_ptr = static_cast<uint8_t *>(MapViewOfFile(_handle, access, 0, 0, _shm_size));

  if (!_shm_ptr || (_inject_error == -4)) {
    PLOG_ERROR << "Mapping of shm failed: name=" << name;
    return false;
  }

  PLOG_INFO << "Opened : " << _shm_name << "  size: " << _shm_size;
  return true;
}

bool CrossPlatformSharedMemory::Close() {
  if (_shm_ptr) {
    UnmapViewOfFile(_shm_ptr);
    _shm_ptr = nullptr;
  }

  if (_handle && _handle != INVALID_HANDLE_VALUE) {
    CloseHandle(_handle);
    _handle = nullptr;
  }

  PLOG_INFO << "Closed : " << _shm_name;
  return true;
}

// For testing
bool CrossPlatformSharedMemory::IsOpen() { return _shm_ptr != nullptr; }

unsigned int CrossPlatformSharedMemory::Size() { return _shm_size; }

uint8_t *CrossPlatformSharedMemory::Data() { return _shm_ptr; }

std::string CrossPlatformSharedMemory::Name() { return _shm_name; }

#else

// Shared memory is based on shm functions.

CrossPlatformSharedMemory::CrossPlatformSharedMemory() {
  _shm_ptr = nullptr;
  _shm_size = 0;
}

CrossPlatformSharedMemory::~CrossPlatformSharedMemory() { Close(); }

bool CrossPlatformSharedMemory::Create(std::string name, int size) {
  // Create shared memory object
  _shm_name = name;
  _shm_size = size;
  _shm_fd = shm_open(_shm_name.c_str(), O_CREAT | O_RDWR,
                     0755); // Create with read/write permission
  if (_shm_fd < 0 || (_inject_error == -1)) {
    PLOG_ERROR << "Failed to create shared memory object: " << _shm_name
               << " error: " << strerror(errno);
    return false;
  }

  // Set size of the shared memory object
  if ((ftruncate(_shm_fd, _shm_size) == -1) || (_inject_error == -2)) {
    PLOG_ERROR << "Failed to set size for shared memory: " << _shm_name << " size: " << _shm_size
               << " error: " << strerror(errno);
    close(_shm_fd);
    _shm_fd = -1;
    return false;
  }

  // Map the shared memory object into the address space
  _shm_ptr = (uint8_t *)mmap(nullptr, _shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, _shm_fd, 0);

  if ((_shm_ptr == MAP_FAILED) || (_inject_error == -3)) {
    PLOG_ERROR << "Failed to map shared memory: " << _shm_name << " error: " << strerror(errno);
    close(_shm_fd);
    _shm_fd = -1;
    return false;
  }

  PLOG_INFO << "Created Shared memory object: " << _shm_name << " with size: " << _shm_size;

  return true;
} // For testing

// Normal use is read-only
bool CrossPlatformSharedMemory::Open(std::string name, int size) {
  // Create shared memory object
  _shm_name = name;
  _shm_size = size;
  _shm_fd = shm_open(_shm_name.c_str(), O_RDONLY,
                     0755); // Create with read/write permission
  if ((_shm_fd < 0) || (_inject_error == -4)) {
    PLOG_ERROR << "Failed to open shared memory object: " << _shm_name
               << " error: " << strerror(errno);
    return false;
  }

  // Map the shared memory object into the address space
  _shm_ptr = (uint8_t *)mmap(nullptr, _shm_size, PROT_READ, MAP_SHARED, _shm_fd, 0);

  if ((_shm_ptr == MAP_FAILED) || (_inject_error == -5)) {
    PLOG_ERROR << "Failed to map shared memory: " << _shm_name << " error: " << strerror(errno);
    close(_shm_fd);
    _shm_fd = -1;
    return false;
  }

  PLOG_INFO << "Open Shared memory object: " << _shm_name;

  return true;
}

bool CrossPlatformSharedMemory::Close() {
  if (_shm_fd < 0) {
    return true; // Nothing to do
  }

  // TODO  revisit this error-handling on failures

  // Unmap the shared memory
  if ((munmap(_shm_ptr, sizeof(_shm_size)) == -1) || (_inject_error == -6)) {
    PLOG_ERROR << "Failed to unmap shared memory: " << strerror(errno);
    close(_shm_fd);
    _shm_fd = -1;
    shm_unlink(_shm_name.c_str());

    _shm_ptr = nullptr;
    return false;
  }

  // Close the shared memory object
  close(_shm_fd);
  _shm_fd = -1;

  // Unlink the shared memory object (this removes the shared memory object
  // from the system)
  if ((shm_unlink(_shm_name.c_str()) == -1) || (_inject_error == -7)) {
    PLOG_ERROR << "Failed to unlink shared memory: " << _shm_name << " error: " << strerror(errno);
    return false;
  }

  PLOG_INFO << "Closed Shared memory object: " << _shm_name;
  return true;

} // When cleaning up

bool CrossPlatformSharedMemory::IsOpen() { return _shm_fd > -1; } // For testing

unsigned int CrossPlatformSharedMemory::Size() { return _shm_size; }

uint8_t *CrossPlatformSharedMemory::Data() { return _shm_ptr; }

std::string CrossPlatformSharedMemory::Name() { return _shm_name; }
#endif
