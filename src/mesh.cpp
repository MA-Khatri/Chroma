#include "mesh.hpp"

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
  //clang-format on
  
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

      glm::vec3 position = {cosPhi * sinTheta, cosTheta, sinPhi * sinTheta};
      glm::vec3 normal = glm::normalize(position);
      glm::vec2 texCoords = {1.0f - (float)lon / longitudeSegments, 1.0f - (float)lat / latitudeSegments};

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
  for (int i = -xCount; i < xCount + 1; i++)
  {
    if (i == 0) continue; // Skip the line through the origin since it will be drawn separately
    
    vertices[index] = {{i * xGap, -groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, xGridColor, {0.0f, 0.0f}};
    indices[index] = index;
    index++;

    vertices[index] = {{i * xGap, groundGridYMax, 0.0f}, {0.0f, 0.0f, 1.0f}, xGridColor, {1.0f, 1.0f}};
    indices[index] = index;
    index++;
  }

  // Lines along y-axis spanning from -groundGridXMax to groundGridXMax
  for (int i = -yCount; i < yCount + 1; i++)
  {
    if (i == 0) continue; // Skip the line through the origin since it will be drawn separately

    vertices[index] = {{-groundGridXMax, i * yGap, 0.0f}, {0.0f, 0.0f, 1.0f}, yGridColor, {0.0f, 0.0f}};
    indices[index] = index;
    index++;

    vertices[index] = {{groundGridXMax, i * yGap, 0.0f}, {0.0f, 0.0f, 1.0f}, yGridColor, {1.0f, 1.0f}};
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