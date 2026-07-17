#pragma once

#include "texture.hpp"

enum class MaterialType { 
  // Surface materials
  Lambertian, 
  Conductor, 
  Dielectric, 
  Principled, 
  Emissive, 
  
  // Special non-surface materials
  Point,
  Lines, 
  GroundGrid, // Special material with custom shaders for the ground grid
};

// Single global counter for assigning unique material IDs
inline int MaterialCounter = 0;

class Material {
public:
  Material(TexturePaths texturePaths, MaterialType type);
  Material(MaterialType type) : m_Type(type) {};
  Material() : m_Type(MaterialType::Lambertian) {};
  ~Material() {};

  int m_MaterialID = MaterialCounter++;

  MaterialType m_Type;

  // Base material properties
  glm::vec3 m_Albedo = glm::vec3(1.0f);
  glm::vec3 m_Emissive = glm::vec3(0.0f);

  float m_AO = 1.0f; // Ambient occlusion

  // Dielectric / Conductor properties
  glm::vec3 m_ReflectionColor = glm::vec3(1.0f);
  glm::vec3 m_RefractionColor = glm::vec3(1.0f);
  glm::vec3 m_Extinction = glm::vec3(0.0f);

  float m_EtaIn = 1.0f;  // Index of refraction inside the material
  float m_EtaOut = 1.0f; // Index of refraction outside the material

  // Principled BSDF material properties
  float m_SpecularTransmission = 0.0f;
  float m_Metallic = 0.0f;
  float m_Subsurface = 0.0f;
  float m_Specular = 0.5f;
  float m_Roughness = 0.5f;
  float m_SpecularTint = 0.0f;
  float m_Anisotropic = 0.0f;
  float m_Sheen = 0.0f;
  float m_SheenTint = 0.5f;
  float m_Clearcoat = 0.0f;
  float m_ClearcoatGloss = 1.0f;

  // Textures
  Texture<uint8_t> m_AlbedoTexture;
  Texture<uint8_t> m_NormalTexture;
  Texture<uint8_t> m_MetallicTexture;
  Texture<uint8_t> m_RoughnessTexture; // Aka "specular" map
  Texture<uint8_t> m_HeightTexture;
  Texture<uint8_t> m_AOTexture;

  // Special non-surface material properties
  float m_PointSize = 1.0f; // For point materials
  float m_LineWidth = 1.0f; // For line materials
};