#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <string>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <vulkan/vulkan.h>


struct Vertex
{
	glm::vec3 posn;
	glm::vec3 normal;
	glm::vec2 texCoord;


	static VkVertexInputBindingDescription GetBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}


	static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, posn);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}


	bool operator==(const Vertex& other) const {
		return posn == other.posn && normal == other.normal && texCoord == other.texCoord;
	}
};

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.posn) ^
				(hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}


struct Mesh
{
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkPrimitiveTopology drawMode = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	float lineWidth = 1.0f;

	std::vector<glm::ivec3> ivecIndices;
	std::vector<glm::vec3> posns;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texCoords;

	/* 
	 * Helper that converts index vector to vector of ivec3 where each represents the three indices for 1 triangle,
	 * and splits up vector of Vertex to vectors of individual components of each Vertex.
	 */
	void SetupOptixMesh()
	{
		if (drawMode == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
		{
			ivecIndices.reserve(indices.size() / 3);
			for (int i = 0; i < indices.size(); i += 3)
			{
				ivecIndices.emplace_back(glm::ivec3(indices[i + 0], indices[i + 1], indices[i + 2]));
			}
		}
		// TODO: index vector conversion to ivec3 for other draw modes (e.g. triangle fan, triangle strip)
		else
		{
			std::cerr << "Index vector conversion to ivec3 not supported for draw mode: " << drawMode << std::endl;
			exit(-1);
		}

		posns.reserve(vertices.size());
		normals.reserve(vertices.size());
		texCoords.reserve(vertices.size());
		for (auto& vertex : vertices)
		{
			posns.emplace_back(vertex.posn);
			normals.emplace_back(vertex.normal);
			texCoords.emplace_back(vertex.texCoord);
		}
	}
};


Mesh CreateHelloTriangle();
Mesh CreatePlane();

/* Create ground plane line grid except for x = 0, y = 0 -- note: Vertex.normal = color of the lines */
Mesh CreateGroundGrid();

/* Create XY axes -- this is separate since we use a diff. pipeline to render them thicker */
Mesh CreateXYAxes(); 

Mesh LoadMesh(std::string filepath);