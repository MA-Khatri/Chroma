#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <string>

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;
  glm::vec2 texCoords;

  bool operator==(const Vertex &other) const {
    return position == other.position && normal == other.normal && color == other.color &&
           texCoords == other.texCoords;
  }
};

template <> struct std::hash<Vertex> {
  size_t operator()(Vertex const &vertex) const {
    size_t h1 = std::hash<glm::vec3>()(vertex.position);
    size_t h2 = std::hash<glm::vec3>()(vertex.normal);
    size_t h3 = std::hash<glm::vec3>()(vertex.color);
    size_t h4 = std::hash<glm::vec2>()(vertex.texCoords);
    // combine position, normal, color and texCoords hashes with XOR and bit shifting
    return (((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1)) ^ (h4 << 2);
  }
};

enum class DrawMode { Points, Lines, LineStrip, LineLoop, Triangles, TriangleStrip, TriangleFan };

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  DrawMode drawMode = DrawMode::Triangles;
  float pointSize = 1.0f;
  float lineWidth = 1.0f;

  Mesh(const std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices, DrawMode drawMode)
      : vertices(vertices), indices(indices), drawMode(drawMode) {}
};

Mesh CreateHelloTriangleMesh();

Mesh CreatePlaneMesh(float width = 1, float depth = 1, int widthSegments = 1,
                     int depthSegments = 1);

Mesh CreateCubeMesh();

Mesh CreateSphereMesh(int latitudeSegments, int longitudeSegments);

Mesh CreateIcosphere(int subdivisions);

Mesh CreateGroundGridMesh();

// Create XY axes separate from ground grid since we render them with a thicker line width
Mesh CreateXYAxesMesh();

// Load a mesh from a file. The file format is determined by the file extension.
Mesh LoadMeshFromFile(const std::string &filepath);
Mesh LoadMeshFromOBJ(const std::string &filepath);
Mesh LoadMeshFromPLY(const std::string &filepath);