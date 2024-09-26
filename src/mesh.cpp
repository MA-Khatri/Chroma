#include "mesh.h"


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

	return mesh;
}