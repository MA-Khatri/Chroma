#pragma once

#include <SDL3/SDL_pixels.h>
#include <SDL3_image/SDL_image.h>
#include <glm/glm.hpp>
#include <plog/Log.h>
#include <string>
#include <vector>

enum class TextureType { Albedo, Normal, Metallic, Roughness, Height, AO };

struct TexturePaths {
  std::string albedo;
  std::string normal;
  std::string metallic;
  std::string roughness;
  std::string height;
  std::string ao; // ambient occlusion
};

template <typename T> struct Texture {
public:
  Texture() = default;
  Texture(const std::string &path, TextureType type) : path(path), type(type) {
    SDL_Surface *surface = IMG_Load(path.c_str());
    if (!surface) {
      PLOG_ERROR << "Failed to load texture: " << path << " SDL_image Error: " << IMG_GetError();
      size = glm::ivec3(0, 0, 0);
      return;
    }

    int bytesPerPixel = SDL_BYTESPERPIXEL(surface->format);
    if (bytesPerPixel != sizeof(T)) {
      PLOG_ERROR << "Unexpected pixel format in texture: " << path
                 << " Expected bytes per pixel: " << sizeof(T)
                 << " Actual bytes per pixel: " << bytesPerPixel;
      SDL_DestroySurface(surface);
      size = glm::ivec3(0, 0, 0);
      return;
    }
    size = glm::ivec3(surface->w, surface->h, bytesPerPixel);

    // Copy pixel data
    pixels.resize(surface->w * surface->h * bytesPerPixel);
    std::memcpy(pixels.data(), surface->pixels, pixels.size() * sizeof(T));

    SDL_DestroySurface(surface);
  }
  ~Texture() = default;

  std::string path;
  TextureType type;

  std::vector<T> pixels;
  glm::ivec3 size; // width, height, channels
};