#pragma once

#include <memory>

#include "material.hpp"
#include "mesh.hpp"
#include "transform.hpp"

class Object {
public:
  Object(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material, Transform transform)
      : mesh(mesh), material(material), transform(transform) {}
  ~Object() = default;

  std::shared_ptr<Mesh> mesh;
  std::shared_ptr<Material> material;
  Transform transform;
};