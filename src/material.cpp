#include "material.hpp"

Material::Material(TexturePaths texturePaths, MaterialType type) : m_Type(type) {
  m_AlbedoTexture = Texture<uint8_t>(texturePaths.albedo, TextureType::Albedo);
  m_NormalTexture = Texture<uint8_t>(texturePaths.normal, TextureType::Normal);
  m_MetallicTexture = Texture<uint8_t>(texturePaths.metallic, TextureType::Metallic);
  m_RoughnessTexture = Texture<uint8_t>(texturePaths.roughness, TextureType::Roughness);
  m_HeightTexture = Texture<uint8_t>(texturePaths.height, TextureType::Height);
  m_AOTexture = Texture<uint8_t>(texturePaths.ao, TextureType::AO);
}

// TODO