#pragma once

#include "texture.hpp"

enum class MaterialType { Lambertian, Conductor, Dielectric, Principled, Emissive };

class Material {
public:
  Material(TexturePaths texturePaths, MaterialType type);
  Material() : type(MaterialType::Lambertian) {}
  ~Material();

  MaterialType type;

  // Base material properties
  glm::vec3 albedo = glm::vec3(1.0f);
  glm::vec3 emissive = glm::vec3(0.0f);

  float ao = 1.0f; // Ambient occlusion

  // Dielectric / Conductor properties
  glm::vec3 reflection_color = glm::vec3(1.0f);
  glm::vec3 refraction_color = glm::vec3(1.0f);
  glm::vec3 extinction = glm::vec3(0.0f);

  float eta_in = 1.0f;  // Index of refraction inside the material
  float eta_out = 1.0f; // Index of refraction outside the material

  // Principled BSDF material properties
  float specularTransmission = 0.0f;
  float metallic = 0.0f;
  float subsurface = 0.0f;
  float specular = 0.5f;
  float roughness = 0.5f;
  float specularTint = 0.0f;
  float anisotropic = 0.0f;
  float sheen = 0.0f;
  float sheenTint = 0.5f;
  float clearcoat = 0.0f;
  float clearcoatGloss = 1.0f;

  // Textures
  Texture<uint8_t> albedoTexture;
  Texture<uint8_t> normalTexture;
  Texture<uint8_t> metallicTexture;
  Texture<uint8_t> roughnessTexture;
  Texture<uint8_t> heightTexture;
  Texture<uint8_t> aoTexture;
};