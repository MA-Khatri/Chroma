#include "material.hpp"

Material::Material(TexturePaths texturePaths, MaterialType type) : m_Type(type) {
  m_AlbedoTexture = Texture<uint8_t>(texturePaths.albedo, TextureType::Albedo);
  // m_NormalTexture = Texture<uint8_t>(texturePaths.normal, TextureType::Normal);
  // m_SpecularTexture = Texture<uint8_t>(texturePaths.specular, TextureType::Specular);
}

// TODO