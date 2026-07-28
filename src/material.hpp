#pragma once

#include "texture.hpp"
#include <cstdint>

enum class MaterialType {
  // Note: "Flat" == no shading
  // Surface materials
  SurfacePerVertexFlat,
  SurfacePerVertexShaded,
  SurfacePerVertexNormal,
  SurfaceAlbedoTextureFlat,
  SurfaceAlbedoTextureShaded,

  // Points
  PointFlat,
  PointShaded,
  PointNormal,

  // Lines
  Lines,
  GroundGrid, // Special material with custom shaders for the ground grid
};

// Single global counter for assigning unique material IDs
inline int MaterialCounter = 0;

class Material {
public:
  Material(TexturePaths texturePaths, MaterialType type);
  Material(MaterialType type) : m_Type(type) {};
  Material() : m_Type(MaterialType::PointFlat) {};
  ~Material() {};

  bool HasTextures() {
    return !m_AlbedoTexture.empty(); // Add more as necessary
  }

  int m_MaterialID = MaterialCounter++;

  MaterialType m_Type;

  // Textures
  Texture<uint8_t> m_AlbedoTexture;
  // Texture<uint8_t> m_NormalTexture;
  // Texture<uint8_t> m_SpecularTexture;

  // Special non-surface material properties
  float m_PointSize = 1.0f; // For point materials
  float m_LineWidth = 1.0f; // For line materials
};