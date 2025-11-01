#pragma once

#include <memory>

#include "material.hpp"
#include "mesh.hpp"
#include "transform.hpp"

class Object {
public:
  Object(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material, Transform transform)
      : m_Mesh(mesh), m_Material(material), m_Transform(transform) {}
  ~Object() = default;

  std::shared_ptr<Mesh> m_Mesh;
  std::shared_ptr<Material> m_Material;
  Transform m_Transform;
};