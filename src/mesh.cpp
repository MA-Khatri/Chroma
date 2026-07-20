#include "mesh.hpp"

#include <fstream>
#include <plog/Log.h>
#include <tiny_obj_loader.h>
#include <tinyply.h>

Mesh CreateHelloTriangleMesh() {
  std::vector<Vertex> vertices = {
      {{0.0f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 1.0f}},
      {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
      {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2};

  return Mesh(vertices, indices, DrawMode::Triangles);
}

Mesh CreatePlaneMesh(float width, float depth, int widthSegments, int depthSegments) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  for (int y = 0; y <= depthSegments; ++y) {
    for (int x = 0; x <= widthSegments; ++x) {
      float xPos = ((float)x / widthSegments - 0.5f) * width;
      float yPos = ((float)y / depthSegments - 0.5f) * depth;
      vertices.push_back({{xPos, yPos, 0.0f},
                          {0.0f, 0.0f, 1.0f},
                          {1.0f, 1.0f, 1.0f},
                          {(float)x / widthSegments, (float)y / depthSegments}});
    }
  }

  for (int y = 0; y < depthSegments; ++y) {
    for (int x = 0; x < widthSegments; ++x) {
      int topLeft = y * (widthSegments + 1) + x;
      int topRight = topLeft + 1;
      int bottomLeft = (y + 1) * (widthSegments + 1) + x;
      int bottomRight = bottomLeft + 1;

      indices.push_back(topLeft);
      indices.push_back(bottomLeft);
      indices.push_back(topRight);

      indices.push_back(topRight);
      indices.push_back(bottomLeft);
      indices.push_back(bottomRight);
    }
  }

  return Mesh(vertices, indices, DrawMode::Triangles);
}

Mesh CreateCubeMesh() {
  // Cube centered at origin with side length = 1 (extents [-0.5, 0.5] on x,y,z).
  // clang-format off
  std::vector<Vertex> vertices = {
      // +X face
      {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
      {{0.5f, -0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
      {{0.5f,  0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
      {{0.5f,  0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
      // -X face
      {{-0.5f, -0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
      {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
      {{-0.5f,  0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f,  0.5f,  0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
      // +Y face
      {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
      {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
      {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
      // -Y face
      {{-0.5f, -0.5f,  0.5f}, {0.0f,-1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
      {{ 0.5f, -0.5f,  0.5f}, {0.0f,-1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
      {{ 0.5f, -0.5f, -0.5f}, {0.0f,-1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f, -0.5f, -0.5f}, {0.0f,-1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
      // +Z face (top)
      {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
      {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
      {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
      // -Z face (bottom)
      {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f,-1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}},
      {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f,-1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f}},
      {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f,-1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
      {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f,-1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
  };
  // clang-format on

  std::vector<uint32_t> indices;
  indices.reserve(36);
  for (uint32_t face = 0; face < 6; ++face) {
    uint32_t base = face * 4;
    indices.push_back(base + 0);
    indices.push_back(base + 1);
    indices.push_back(base + 2);
    indices.push_back(base + 2);
    indices.push_back(base + 3);
    indices.push_back(base + 0);
  }

  return Mesh(vertices, indices, DrawMode::Triangles);
}

Mesh CreateSphereMesh(int latitudeSegments, int longitudeSegments) {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  for (int lat = 0; lat <= latitudeSegments; ++lat) {
    float theta = lat * glm::pi<float>() / latitudeSegments;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);

    for (int lon = 0; lon <= longitudeSegments; ++lon) {
      float phi = lon * 2.0f * glm::pi<float>() / longitudeSegments;
      float sinPhi = sin(phi);
      float cosPhi = cos(phi);

      glm::vec3 position = {cosPhi * sinTheta, sinPhi * sinTheta, cosTheta};
      glm::vec3 normal = glm::normalize(position);
      glm::vec2 texCoords = {1.0f - (float)lon / longitudeSegments,
                             1.0f - (float)lat / latitudeSegments};

      vertices.push_back({position, normal, {1.0f, 1.0f, 1.0f}, texCoords});
    }
  }

  for (int lat = 0; lat < latitudeSegments; ++lat) {
    for (int lon = 0; lon < longitudeSegments; ++lon) {
      int first = (lat * (longitudeSegments + 1)) + lon;
      int second = first + longitudeSegments + 1;

      indices.push_back(first);
      indices.push_back(second);
      indices.push_back(first + 1);

      indices.push_back(second);
      indices.push_back(second + 1);
      indices.push_back(first + 1);
    }
  }

  return Mesh(vertices, indices, DrawMode::Triangles);
}

Mesh CreateIcosphere(int subdivisions) {
  // TODO: Create an icosahedron and then subdivide it to create an icosphere
  // This is a placeholder implementation; a full implementation would require more code
  // For now, we can return a simple sphere mesh as a placeholder
  return CreateSphereMesh(10, 10);
}

static const float groundGridXMax = 500.0f;
static const float groundGridYMax = 500.0f;

Mesh CreateGroundGridMesh() {
  const glm::vec3 xGridColor = glm::vec3(78.0f / 255.0f, 78.0f / 255.0f, 78.0f / 255.0f);
  const glm::vec3 yGridColor = glm::vec3(78.0f / 255.0f, 78.0f / 255.0f, 78.0f / 255.0f);

  // Count from 0 to +x/y max -- actual grid extends x/yCount in pos/neg directions
  const int xCount = 500;
  const int yCount = 500;
  const int numVertices = xCount * 4 + yCount * 4;

  const float xGap = groundGridXMax / xCount;
  const float yGap = groundGridYMax / yCount;

  std::vector<Vertex> vertices(numVertices);
  std::vector<uint32_t> indices(numVertices);

  // Lines along x-axis spanning from -groundGridYMax to groundGridYMax
  int index = 0;
  for (int i = -xCount; i < xCount + 1; i++) {
    if (i == 0)
      continue; // Skip the line through the origin since it will be drawn separately

    vertices[index] = {
        {i * xGap, -groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, xGridColor, {0.0f, 0.0f}};
    indices[index] = index;
    index++;

    vertices[index] = {
        {i * xGap, groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, xGridColor, {1.0f, 1.0f}};
    indices[index] = index;
    index++;
  }

  // Lines along y-axis spanning from -groundGridXMax to groundGridXMax
  for (int i = -yCount; i < yCount + 1; i++) {
    if (i == 0)
      continue; // Skip the line through the origin since it will be drawn separately

    vertices[index] = {
        {-groundGridXMax, i * yGap, 0.0f}, {0.0f, 0.0f, 1.0f}, yGridColor, {0.0f, 0.0f}};
    indices[index] = index;
    index++;

    vertices[index] = {
        {groundGridXMax, i * yGap, 0.0f}, {0.0f, 0.0f, 1.0f}, yGridColor, {1.0f, 1.0f}};
    indices[index] = index;
    index++;
  }

  return Mesh(vertices, indices, DrawMode::Lines);
}

Mesh CreateXYAxesMesh() {
  const glm::vec3 xAxisColor = glm::vec3(98.0f / 255.0f, 135.0f / 255.0f, 41.0f / 255.0f);
  const glm::vec3 yAxisColor = glm::vec3(154.0f / 255.0f, 60.0f / 255.0f, 74.0f / 255.0f);

  std::vector<Vertex> vertices = {
      // X axis
      {{-groundGridXMax, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, xAxisColor, {0.0f, 0.0f}},
      {{groundGridXMax, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, xAxisColor, {1.0f, 1.0f}},
      // Y axis
      {{0.0f, -groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, yAxisColor, {0.0f, 0.0f}},
      {{0.0f, groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, yAxisColor, {1.0f, 1.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2, 3};

  return Mesh(vertices, indices, DrawMode::Lines);
}

Mesh LoadMeshFromFile(const std::string &filepath) {
  // Check the file extension to determine the loader to use
  std::string extension = filepath.substr(filepath.find_last_of(".") + 1);

  if (extension == "obj") {
    return LoadMeshFromOBJ(filepath);
  } else if (extension == "ply") {
    return LoadMeshFromPLY(filepath);
  } else {
    PLOG_ERROR << "Unsupported mesh file format: " << extension;
    return Mesh();
  }
}

Mesh LoadMeshFromOBJ(const std::string &filepath) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  std::string baseDir = filepath.substr(0, filepath.find_last_of("/\\") + 1);

  bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(),
                             baseDir.c_str());

  if (!warn.empty()) {
    PLOG_WARNING << warn;
  }

  if (!ok) {
    PLOG_ERROR << "Failed to load OBJ file: " << filepath << " (" << err << ")";
    return Mesh();
  }

  // Detect point cloud: no shape produced any face indices, but vertices exist.
  size_t totalFaceIndices = 0;
  for (const auto &shape : shapes) {
    totalFaceIndices += shape.mesh.indices.size();
  }
  bool isPointCloud = (totalFaceIndices == 0) && !attrib.vertices.empty();

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  bool hasColors = !attrib.colors.empty();

  if (isPointCloud) {
    // No face/index data available — one Vertex per raw position, in file order.
    size_t vertexCount = attrib.vertices.size() / 3;
    vertices.resize(vertexCount);
    indices.resize(vertexCount);

    for (size_t i = 0; i < vertexCount; ++i) {
      Vertex &vertex = vertices[i];
      vertex.position = {
          attrib.vertices[3 * i + 0],
          attrib.vertices[3 * i + 1],
          attrib.vertices[3 * i + 2],
      };

      if (!attrib.normals.empty()) {
        vertex.normal = {
            attrib.normals[3 * i + 0],
            attrib.normals[3 * i + 1],
            attrib.normals[3 * i + 2],
        };
      }

      vertex.color = hasColors ? glm::vec3(attrib.colors[3 * i + 0], attrib.colors[3 * i + 1],
                                           attrib.colors[3 * i + 2])
                               : glm::vec3(1.0f);

      indices[i] = static_cast<uint32_t>(i);
    }

    PLOG_INFO << "Detected point cloud OBJ (" << vertexCount << " points): " << filepath;
    return Mesh(vertices, indices, DrawMode::Points);
  }

  // Regular triangle mesh path — dedupe vertices by full attribute set.
  std::unordered_map<Vertex, uint32_t> uniqueVertices;

  for (const auto &shape : shapes) {
    for (const auto &index : shape.mesh.indices) {
      Vertex vertex{};

      vertex.position = {
          attrib.vertices[3 * index.vertex_index + 0],
          attrib.vertices[3 * index.vertex_index + 1],
          attrib.vertices[3 * index.vertex_index + 2],
      };

      if (index.normal_index >= 0) {
        vertex.normal = {
            attrib.normals[3 * index.normal_index + 0],
            attrib.normals[3 * index.normal_index + 1],
            attrib.normals[3 * index.normal_index + 2],
        };
      }

      vertex.color = hasColors ? glm::vec3(attrib.colors[3 * index.vertex_index + 0],
                                           attrib.colors[3 * index.vertex_index + 1],
                                           attrib.colors[3 * index.vertex_index + 2])
                               : glm::vec3(1.0f);

      if (index.texcoord_index >= 0) {
        vertex.texCoords = {
            attrib.texcoords[2 * index.texcoord_index + 0],
            attrib.texcoords[2 * index.texcoord_index + 1],
        };
      } else {
        vertex.texCoords = {0.0f, 0.0f};
      }

      auto it = uniqueVertices.find(vertex);
      if (it == uniqueVertices.end()) {
        uint32_t newIndex = static_cast<uint32_t>(vertices.size());
        uniqueVertices.emplace(vertex, newIndex);
        vertices.push_back(vertex);
        indices.push_back(newIndex);
      } else {
        indices.push_back(it->second);
      }
    }
  }

  return Mesh(vertices, indices, DrawMode::Triangles);
}

Mesh LoadMeshFromPLY(const std::string &filepath) {
  using namespace tinyply;

  std::ifstream fileStream(filepath, std::ios::binary);
  if (!fileStream || fileStream.fail()) {
    PLOG_ERROR << "Failed to open PLY file: " << filepath;
    return Mesh();
  }

  PlyFile file;
  file.parse_header(fileStream);

  // Detect point cloud: no "face" element declared in the header, or it's empty.
  bool isPointCloud = true;
  for (const auto &element : file.get_elements()) {
    if (element.name == "face" && element.size > 0) {
      isPointCloud = false;
      break;
    }
  }

  std::shared_ptr<PlyData> plyPositions, plyNormals, plyColors, plyTexCoords, plyFaces;

  try {
    plyPositions = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    PLOG_ERROR << "PLY file missing vertex positions: " << e.what();
    return Mesh();
  }

  try {
    plyNormals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
  } catch (const std::exception &) {
    PLOG_WARNING << "PLY file missing vertex normals, proceeding without normals.";
  }

  try {
    plyColors = file.request_properties_from_element("vertex", {"red", "green", "blue"});
  } catch (const std::exception &) {
    try {
      plyColors = file.request_properties_from_element("vertex", {"r", "g", "b"});
    } catch (const std::exception &) {
      PLOG_WARNING << "PLY file missing vertex colors, proceeding without colors.";
    }
  }

  try {
    plyTexCoords = file.request_properties_from_element("vertex", {"u", "v"});
  } catch (const std::exception &) {
    try {
      plyTexCoords = file.request_properties_from_element("vertex", {"s", "t"});
    } catch (const std::exception &) {
      PLOG_WARNING << "PLY file missing vertex texture coordinates, proceeding without texCoords.";
    }
  }

  if (!isPointCloud) {
    try {
      plyFaces = file.request_properties_from_element("face", {"vertex_indices"}, 3);
    } catch (const std::exception &e) {
      // Header claimed faces but tinyply couldn't bind them — fall back to points.
      PLOG_WARNING << "PLY face element present but unreadable, treating as point cloud: "
                   << e.what();
      isPointCloud = true;
    }
  }

  file.read(fileStream);

  size_t vertexCount = plyPositions->count;
  std::vector<Vertex> vertices(vertexCount);

  auto readVec3 = [](const std::shared_ptr<PlyData> &data, size_t count, auto assign) {
    if (data->t == Type::FLOAT32) {
      const float *p = reinterpret_cast<const float *>(data->buffer.get());
      for (size_t i = 0; i < count; ++i)
        assign(i, p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    } else if (data->t == Type::FLOAT64) {
      const double *p = reinterpret_cast<const double *>(data->buffer.get());
      for (size_t i = 0; i < count; ++i)
        assign(i, p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    }
  };

  readVec3(plyPositions, vertexCount, [&](size_t i, double x, double y, double z) {
    vertices[i].position = glm::vec3(x, y, z);
  });

  if (plyNormals) {
    readVec3(plyNormals, vertexCount, [&](size_t i, double x, double y, double z) {
      vertices[i].normal = glm::vec3(x, y, z);
    });
  }

  if (plyColors) {
    if (plyColors->t == Type::UINT8) {
      const uint8_t *p = reinterpret_cast<const uint8_t *>(plyColors->buffer.get());
      for (size_t i = 0; i < vertexCount; ++i) {
        vertices[i].color =
            glm::vec3(p[3 * i] / 255.0f, p[3 * i + 1] / 255.0f, p[3 * i + 2] / 255.0f);
      }
    } else {
      readVec3(plyColors, vertexCount, [&](size_t i, double r, double g, double b) {
        vertices[i].color = glm::vec3(r, g, b);
      });
    }
  } else {
    for (auto &v : vertices)
      v.color = glm::vec3(1.0f);
  }

  if (plyTexCoords) {
    if (plyTexCoords->t == Type::FLOAT32) {
      const float *p = reinterpret_cast<const float *>(plyTexCoords->buffer.get());
      for (size_t i = 0; i < vertexCount; ++i)
        vertices[i].texCoords = glm::vec2(p[2 * i], p[2 * i + 1]);
    } else if (plyTexCoords->t == Type::FLOAT64) {
      const double *p = reinterpret_cast<const double *>(plyTexCoords->buffer.get());
      for (size_t i = 0; i < vertexCount; ++i)
        vertices[i].texCoords = glm::vec2(p[2 * i], p[2 * i + 1]);
    }
  } else {
    for (auto &v : vertices)
      v.texCoords = glm::vec2(0.0f, 0.0f);
  }

  if (isPointCloud) {
    std::vector<uint32_t> indices(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i)
      indices[i] = static_cast<uint32_t>(i);

    PLOG_INFO << "Detected point cloud PLY (" << vertexCount << " points): " << filepath;
    return Mesh(vertices, indices, DrawMode::Points);
  }

  std::vector<uint32_t> indices;
  indices.reserve(plyFaces->count * 3);

  switch (plyFaces->t) {
  case Type::UINT32:
  case Type::INT32: {
    const uint32_t *p = reinterpret_cast<const uint32_t *>(plyFaces->buffer.get());
    indices.assign(p, p + plyFaces->count * 3);
    break;
  }
  case Type::UINT16:
  case Type::INT16: {
    const uint16_t *p = reinterpret_cast<const uint16_t *>(plyFaces->buffer.get());
    for (size_t i = 0; i < plyFaces->count * 3; ++i)
      indices.push_back(p[i]);
    break;
  }
  case Type::UINT8:
  case Type::INT8: {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(plyFaces->buffer.get());
    for (size_t i = 0; i < plyFaces->count * 3; ++i)
      indices.push_back(p[i]);
    break;
  }
  default:
    PLOG_ERROR << "Unsupported PLY face index type: " << filepath;
    return Mesh();
  }

  return Mesh(vertices, indices, DrawMode::Triangles);
}