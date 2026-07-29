#pragma once

#include <memory>

#include "material.hpp"
#include "mesh.hpp"
#include "transform.hpp"

class Object {
public:
  Object(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material,
         std::shared_ptr<Transform> transform)
      : m_Mesh(mesh), m_Material(material), m_Transform(transform) {}
  Object(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material)
      : m_Mesh(mesh), m_Material(material), m_Transform(std::make_shared<Transform>()) {};
  ~Object() = default;

  std::shared_ptr<Mesh> m_Mesh;
  std::shared_ptr<Material> m_Material;
  std::shared_ptr<Transform> m_Transform;

  bool m_Active = true;
};