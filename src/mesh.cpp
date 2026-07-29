#include "mesh.hpp"

#include "sharing/shared_memory_model_layout.hpp"
#include "sharing/wait_until.hpp"

#include <cstdint>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <plog/Log.h>
#include <tiny_obj_loader.h>
#include <tinyply.h>

std::shared_ptr<Mesh> CreateHelloTriangleMesh() {
  std::vector<Vertex> vertices = {
      {{0.0f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 1.0f}},
      {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
      {{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2};

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Triangles);
}

std::shared_ptr<Mesh> CreatePlaneMesh(float width, float depth, int widthSegments,
                                      int depthSegments) {
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

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Triangles);
}

std::shared_ptr<Mesh> CreateCubeMesh() {
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

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Triangles);
}

std::shared_ptr<Mesh> CreateSphereMesh(int latitudeSegments, int longitudeSegments) {
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

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Triangles);
}

std::shared_ptr<Mesh> CreateIcosphere(int subdivisions) {
  // TODO: Create an icosahedron and then subdivide it to create an icosphere
  // This is a placeholder implementation; a full implementation would require more code
  // For now, we can return a simple sphere sphere as a placeholder
  return CreateSphereMesh(10, 10);
}

static const float groundGridXMax = 500.0f;
static const float groundGridYMax = 500.0f;

std::shared_ptr<Mesh> CreateGroundGridMesh() {
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
        {i * xGap, groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, xGridColor, {0.0f, 0.0f}};
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
        {groundGridXMax, i * yGap, 0.0f}, {0.0f, 0.0f, 1.0f}, yGridColor, {0.0f, 0.0f}};
    indices[index] = index;
    index++;
  }

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Lines);
}

std::shared_ptr<Mesh> CreateXYAxesMesh() {
  const glm::vec3 xAxisColor = glm::vec3(98.0f / 255.0f, 135.0f / 255.0f, 41.0f / 255.0f);
  const glm::vec3 yAxisColor = glm::vec3(154.0f / 255.0f, 60.0f / 255.0f, 74.0f / 255.0f);

  std::vector<Vertex> vertices = {
      // X axis
      {{-groundGridXMax, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, xAxisColor, {0.0f, 0.0f}},
      {{groundGridXMax, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, xAxisColor, {0.0f, 0.0f}},
      // Y axis
      {{0.0f, -groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, yAxisColor, {0.0f, 0.0f}},
      {{0.0f, groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, yAxisColor, {0.0f, 0.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2, 3};

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Lines);
}

std::shared_ptr<Mesh> CreateOrientationGizmo() {
  glm::vec3 red = glm::vec3(1.0f, 0.0f, 0.0f);
  glm::vec3 green = glm::vec3(0.0f, 1.0f, 0.0f);
  glm::vec3 blue = glm::vec3(0.0f, 0.0f, 1.0f);

  const float base = 0.7f;
  const float offb = 0.5f;
  glm::vec3 offRed = glm::vec3(base, offb, offb);
  glm::vec3 offGreen = glm::vec3(offb, base, offb);
  glm::vec3 offBlue = glm::vec3(offb, offb, base);

  std::vector<Vertex> vertices = {
      // -x to 0
      {{-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, offRed, {0.0f, 0.0f}},
      {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, offRed, {0.0f, 0.0f}},
      // 0 to +x
      {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, red, {0.0f, 0.0f}},
      {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, red, {0.0f, 0.0f}},

      // -y to 0
      {{0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, offGreen, {0.0f, 0.0f}},
      {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, offGreen, {0.0f, 0.0f}},
      // 0 to +y
      {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, green, {0.0f, 0.0f}},
      {{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, green, {0.0f, 0.0f}},

      // -z to 0
      {{0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, offBlue, {0.0f, 0.0f}},
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, offBlue, {0.0f, 0.0f}},
      // 0 to +z
      {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, blue, {0.0f, 0.0f}},
      {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, blue, {0.0f, 0.0f}},
  };

  std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Lines);
}

std::shared_ptr<Mesh> LoadMeshFromFile(const std::string &filepath) {
  // Check the file extension to determine the loader to use
  std::string extension = filepath.substr(filepath.find_last_of(".") + 1);

  try {
    if (extension == "obj") {
      return LoadMeshFromOBJ(filepath);
    } else if (extension == "ply") {
      return LoadMeshFromPLY(filepath);
    } else {
      PLOG_ERROR << "Unsupported mesh file format: " << extension;
      return nullptr;
    }
  } catch (const std::exception &e) {
    PLOG_ERROR << "Failed loading file " << filepath << ": " << e.what();
    return nullptr;
  }
}

std::shared_ptr<Mesh> LoadMeshFromOBJ(const std::string &filepath) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  std::string search_dir = RES_DIR;
  search_dir += "meshes/";
  std::string fullpath = search_dir + filepath;

  std::string baseDir = fullpath.substr(0, fullpath.find_last_of("/\\") + 1);
  bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, fullpath.c_str(),
                             baseDir.c_str());

  if (!ok) {
    // Try loading from the provided path directly if the relative path fails
    baseDir = filepath.substr(0, filepath.find_last_of("/\\") + 1);
    ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str(),
                          baseDir.c_str());

    if (!ok) {
      PLOG_ERROR << "Failed to load OBJ file: " << filepath << " (" << err << ")";
      return nullptr;
    }
  }

  if (!warn.empty()) {
    PLOG_WARNING << warn;
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
    return std::make_shared<Mesh>(vertices, indices, DrawMode::Points);
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

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Triangles);
}

std::shared_ptr<Mesh> LoadMeshFromPLY(const std::string &filepath) {
  using namespace tinyply;

  std::string search_dir = RES_DIR;
  search_dir += "meshes/";
  std::string fullpath = search_dir + filepath;

  std::ifstream fileStream(fullpath, std::ios::binary);
  if (!fileStream || fileStream.fail()) {
    // Try loading from the provided path directly if the relative path fails
    fileStream.open(filepath, std::ios::binary);
    if (!fileStream || fileStream.fail()) {
      PLOG_ERROR << "Failed to open PLY file: " << fullpath;
      return nullptr;
    }
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
    return nullptr;
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
    return std::make_shared<Mesh>(vertices, indices, DrawMode::Points);
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
    return nullptr;
  }

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Triangles);
}

constexpr uint16_t MODEL_STATE_LOCKED = static_cast<uint16_t>(1 << 5);
std::shared_ptr<Mesh> LoadPointCloudFromSharedMemory(uint8_t *shm, uint32_t &revisionNumber,
                                                     uint32_t &tracking, glm::mat4 &pose) {
  uint32_t oldRevisionNumber = revisionNumber;

  SharedMemoryModelLayout layout(0, 0, shm);

  std::atomic<uint32_t> *busyAtomic(
      (std::atomic<uint32_t> *)((uint32_t *)layout.writing_flag_ptr()));

  bool flagAcquired = wait_until([&]() {
    uint32_t flag0 = 0U;
    uint32_t flag1 = 1U;
    return busyAtomic->compare_exchange_strong(flag0, flag1);
  });

  if (!flagAcquired) {
    uint32_t flag = ((uint32_t *)layout.writing_flag_ptr())[0];
    PLOG_WARNING << "Writing flag is " << flag
                 << ", indicating that model SHM is being written to. Returning nullptr.";
    return nullptr;
  }

  // Get revision number and check if shm has updated
  revisionNumber = ((uint32_t *)layout.revision_ptr())[0];
  if (oldRevisionNumber == revisionNumber) {
    PLOG_VERBOSE << "Revision number is the same as before: " << revisionNumber
                 << ". No update to the point cloud in model SHM. Returning empty nullptr.";

    uint32_t flag0 = 0U;
    memcpy(layout.writing_flag_ptr(), &flag0, sizeof(flag0));
    return nullptr;
  }

  PLOG_VERBOSE << "Acquired new model revision number: " << revisionNumber;

  tracking = ((uint32_t *)layout.tracking_ptr())[0];
  PLOG_VERBOSE << "Tracking: " << tracking;

  // Get model size
  uint32_t totalPoints = ((uint32_t *)layout.n_points_ptr())[0];
  uint32_t nModels = ((uint32_t *)layout.n_models_ptr())[0];
  layout.UpdateSize(totalPoints, nModels);

  PLOG_VERBOSE << "Total points: " << totalPoints;

  // Get pose matrix
  float p[16];
  memcpy(&p, layout.pose_ptr(), 16 * sizeof(float));
  pose = glm::make_mat4(p);

  std::vector<uint32_t> modelSizes(nModels);
  memcpy(modelSizes.data(), layout.model_sizes_ptr(), nModels * sizeof(uint32_t));

  // Memcpy point cloud data
  std::vector<glm::vec3> positions(totalPoints);
  std::vector<glm::vec3> normals(totalPoints);
  std::vector<glm::vec3> colors(totalPoints);
  std::vector<uint16_t> states(totalPoints);

  memcpy(positions.data(), layout.positions_ptr(), totalPoints * layout.single_position_size);
  memcpy(normals.data(), layout.normals_ptr(), totalPoints * layout.single_normal_size);
  memcpy(colors.data(), layout.colors_ptr(), totalPoints * layout.single_color_size);
  memcpy(states.data(), layout.states_ptr(), totalPoints * layout.single_state_size);

  PLOG_VERBOSE << "Completed memcpy of data";

  // Double check model size to see if the data has been updated
  if (totalPoints != ((uint32_t *)layout.n_points_ptr())[0]) {
    PLOG_ERROR << "Model size mismatch: expected " << totalPoints << ", got "
               << ((uint32_t *)layout.n_points_ptr())[0];

    return nullptr;
  }

  uint32_t flag0 = 0U;
  memcpy(layout.writing_flag_ptr(), &flag0, sizeof(flag0));

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  unsigned int filteredCount = 0;
  for (int i = 0; i < totalPoints; i++) {
    auto &p = positions[i];
    auto &n = normals[i];
    auto c = colors[i] / 255.0f;
    auto s = states[i];

    // Skip invalid points
    if (glm::any(glm::isnan(p)) || glm::any(glm::isnan(n)))
      continue;

    bool firstIsland = i < modelSizes[0];
    if (!firstIsland)
      c *= 0.5f;

    Vertex vertex;
    vertex.position = p;
    vertex.normal = n;
    vertex.color = (s & MODEL_STATE_LOCKED) ? glm::vec3(0, 1, 0) : c;

    vertex.texCoords = glm::vec2(0, 0);

    vertices.push_back(vertex);
    indices.push_back(filteredCount++);
  }

  return std::make_shared<Mesh>(vertices, indices, DrawMode::Points);
}