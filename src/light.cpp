#include "light.h"

AreaLight::AreaLight(Vertex v0, Vertex v1, Vertex v2, glm::vec3 radiantExitance, glm::mat4 transform /* = glm::mat4(1.0f)*/, glm::mat4 nTransform /* = glm::mat4(1.0f)*/)
{
	/* === Apply transformations === */
	glm::vec3 p0 = transform * glm::vec4(v0.posn, 1.0f);
	glm::vec3 p1 = transform * glm::vec4(v1.posn, 1.0f);
	glm::vec3 p2 = transform * glm::vec4(v2.posn, 1.0f);

	glm::vec3 n0 = nTransform * glm::vec4(v0.normal, 0.0f);
	glm::vec3 n1 = nTransform * glm::vec4(v1.normal, 0.0f);
	glm::vec3 n2 = nTransform * glm::vec4(v2.normal, 0.0f);


	m_V0 = { ToFloat3(p0), ToFloat3(n0), ToFloat2(v0.texCoord) };
	m_V1 = { ToFloat3(p1), ToFloat3(n1), ToFloat2(v1.texCoord) };
	m_V2 = { ToFloat3(p2), ToFloat3(n2), ToFloat2(v2.texCoord) };


	glm::vec3 ab = p1 - p0;
	glm::vec3 ac = p2 - p0;
	m_Area = 0.5f * glm::length(glm::cross(ab, ac));

	m_Power = m_Area * glm::length(radiantExitance);
	m_LightType = LIGHT_TYPE_AREA;
}


std::vector<std::shared_ptr<Light>> ObjectLight(Object object)
{
	const auto& mesh = object.m_Mesh;
	const auto& indices = mesh->ivecIndices;
	const auto& vertices = mesh->vertices;

	std::vector<std::shared_ptr<Light>> lights;
	lights.reserve(indices.size());

	for (int i = 0; i < indices.size(); i++)
	{
		const auto& cIndices = indices[i];
		lights.push_back(std::make_shared<AreaLight>(vertices[cIndices.x], vertices[cIndices.y], vertices[cIndices.z], object.m_Material->m_EmissionColor, object.m_ModelMatrix, object.m_ModelNormalMatrix));
	}

	return lights;
}