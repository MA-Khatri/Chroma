#pragma once

#include <SDL3/SDL_pixels.h>
#include <SDL3_image/SDL_image.h>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <plog/Log.h>
#include <string>
#include <vector>

enum class TextureType { Albedo, Normal, Metallic, Roughness, Height, AO };

struct TexturePaths { // TODO: should these be stored as std::filesystem::path?
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
  Texture(const std::string &path, TextureType type) : m_FilePath(path), m_Type(type) {
    if (path.empty()) {
      PLOG_VERBOSE << "No texture path provided for texture type " << static_cast<int>(type)
                   << ", skipping load.";
      m_Size = glm::ivec3(0, 0, 0);
      return;
    }

    SDL_Surface *surface = IMG_Load(path.c_str());
    if (!surface) {
      PLOG_ERROR << "Failed to load texture: " << path << " SDL_image Error: " << SDL_GetError();
      m_Size = glm::ivec3(0, 0, 0);
      return;
    }

    int bytesPerPixel = SDL_BYTESPERPIXEL(surface->format);
    if (bytesPerPixel % sizeof(T) != 0) {
      PLOG_ERROR << "Unexpected pixel format in texture: " << path
                 << " Bytes per pixel: " << bytesPerPixel << ", sizeof(T): " << sizeof(T);
      SDL_DestroySurface(surface);
      m_Size = glm::ivec3(0, 0, 0);
      return;
    }
    m_Size = glm::ivec3(surface->w, surface->h, bytesPerPixel);

    // Copy pixel data
    m_Pixels.resize(surface->w * surface->h * bytesPerPixel);
    std::memcpy(m_Pixels.data(), surface->pixels, m_Pixels.size() * sizeof(T));

    SDL_DestroySurface(surface);
  }
  ~Texture() = default;

  std::string m_FilePath;
  TextureType m_Type;

  std::vector<T> m_Pixels;
  glm::ivec3 m_Size; // width, height, bytes per pixel (e.g. 3 for RGB, 4 for RGBA)
};