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
	mesh.SetIndicesIVec3();

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
	mesh.SetIndicesIVec3();

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

			vertex.pos = {
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
	mesh.SetIndicesIVec3();

	return mesh;
}