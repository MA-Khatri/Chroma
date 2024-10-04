#include "mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

Mesh CreateHelloTriangle()
{
	Mesh mesh;

	mesh.vertices = {
		Vertex {glm::vec3( 0.0f, -0.5f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
		Vertex {glm::vec3( 0.5f,  0.5f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(0.0f, 0.0f)},
		Vertex {glm::vec3(-0.5f,  0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(0.0f, 0.0f)},
	};

	mesh.indices = {
		0, 1, 2
	};

	mesh.drawMode = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	mesh.SetupOptixMesh();

	return mesh;
}


Mesh CreatePlane()
{
	Mesh mesh;

	mesh.vertices = {
		Vertex {glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(1.0f, 0.0f)},
		Vertex {glm::vec3( 0.5f, -0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(0.0f, 0.0f)},
		Vertex {glm::vec3( 0.5f,  0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(0.0f, 1.0f)},
		Vertex {glm::vec3(-0.5f,  0.5f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec2(1.0f, 1.0f)},
	};

	mesh.indices = {
		0, 1, 2, 2, 3, 0
	};

	mesh.drawMode = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	mesh.SetupOptixMesh();

	return mesh;
}


Mesh CreateGroundGrid()
{
	/* Axis colors for center lines */
	glm::vec3 xAxisColor = glm::vec3(98.0f / 255.0f, 135.0f / 255.0f, 41.0f / 255.0f);
	glm::vec3 yAxisColor = glm::vec3(154.0f / 255.0f, 60.0f / 255.0f, 74.0f / 255.0f);

	/* Axis color for remaining grid lines */
	glm::vec3 xGridColor = glm::vec3(78.0f / 255.0f, 78.0f / 255.0f, 78.0f / 255.0f);
	glm::vec3 yGridColor = glm::vec3(78.0f / 255.0f, 78.0f / 255.0f, 78.0f / 255.0f);

	/* Count from xgap to +x/+y */
	int xcount = 500;
	int ycount = 500;

	/* Spacing between lines along each axis */
	float xgap = 1.0f;
	float ygap = 1.0f;


	float xmax = xcount * xgap;
	float ymax = ycount * ygap;

	Mesh mesh;
	mesh.vertices.reserve(xcount * 2 + ycount * 2 + 2);
	mesh.indices.reserve(xcount * 2 + ycount * 2 + 2);

	/* Lines along x axis spanning -ymax to +ymax */
	int index = 0;
	for (int i = -xcount; i < xcount + 1; i++)
	{
		Vertex v0;
		v0.posn = glm::vec3(i * xgap, -ymax, 0.0f);
		v0.normal = i == 0 ? xAxisColor : xGridColor;
		v0.texCoord = glm::vec2(0.0f);

		mesh.vertices.emplace_back(v0);
		mesh.indices.emplace_back(index);
		index++;

		Vertex v1;
		v1.posn = glm::vec3(i * xgap, ymax, 0.0f);
		v1.normal = i == 0 ? xAxisColor : xGridColor;
		v1.texCoord = glm::vec2(0.0f);

		mesh.vertices.emplace_back(v1);
		mesh.indices.emplace_back(index);
		index++;
	}
	
	/* Lines along y axis spanning -xmax to +xmax */
	for (int j = -ycount; j < ycount + 1; j++)
	{
		Vertex v0;
		v0.posn = glm::vec3(-xmax, j * ygap, 0.0f);
		v0.normal = j == 0 ? yAxisColor : yGridColor;
		v0.texCoord = glm::vec2(0.0f);

		mesh.vertices.emplace_back(v0);
		mesh.indices.emplace_back(index);
		index++;

		Vertex v1;
		v1.posn = glm::vec3(xmax, j * ygap, 0.0f);
		v1.normal = j == 0 ? yAxisColor : yGridColor;
		v1.texCoord = glm::vec2(0.0f);

		mesh.vertices.emplace_back(v1);
		mesh.indices.emplace_back(index);
		index++;
	}

	mesh.drawMode = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	//mesh.SetupOptixMesh();

	return mesh;
}


Mesh LoadMesh(std::string filepath)
{
	Mesh mesh;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str()))
	{
		std::cerr << warn << err << std::endl;
		exit(-1);
	}


	std::unordered_map<Vertex, uint32_t> uniqueVertices{};

	for (const auto& shape : shapes)
	{
		for (const auto& index : shape.mesh.indices)
		{
			Vertex vertex{};

			vertex.posn = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.texCoord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1] /* flip v-coord to match Vulkan convention */
			};
			
			vertex.normal = {
				attrib.normals[3 * index.normal_index + 0],
				attrib.normals[3 * index.normal_index + 1],
				attrib.normals[3 * index.normal_index + 2]
			};

			if (uniqueVertices.count(vertex) == 0)
			{
				uniqueVertices[vertex] = static_cast<uint32_t>(mesh.vertices.size());
				mesh.vertices.push_back(vertex);
			}

			mesh.indices.push_back(uniqueVertices[vertex]);
		}
	}

	mesh.drawMode = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	mesh.SetupOptixMesh();

	return mesh;
}