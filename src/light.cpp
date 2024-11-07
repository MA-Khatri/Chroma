#include "light.h"

AreaLight::AreaLight(Vertex v0, Vertex v1, Vertex v2, glm::vec3 radiantExitance, glm::mat4 transform /* = glm::mat4(1.0f)*/, glm::mat4 nTransform /* = glm::mat4(1.0f)*/)
{
	m_MISLight.type = LIGHT_TYPE_AREA;

	/* === Pre-apply transformations === */
	m_MISLight.p0 = ToFloat3(glm::vec3(transform * glm::vec4(v0.posn, 1.0f)));
	m_MISLight.p1 = ToFloat3(glm::vec3(transform * glm::vec4(v1.posn, 1.0f)));
	m_MISLight.p2 = ToFloat3(glm::vec3(transform * glm::vec4(v2.posn, 1.0f)));


	m_MISLight.n0 = ToFloat3(glm::vec3(nTransform * glm::vec4(v0.normal, 0.0f)));
	m_MISLight.n1 = ToFloat3(glm::vec3(nTransform * glm::vec4(v1.normal, 0.0f)));
	m_MISLight.n2 = ToFloat3(glm::vec3(nTransform * glm::vec4(v2.normal, 0.0f)));

	m_MISLight.t0 = ToFloat2(v0.texCoord);
	m_MISLight.t1 = ToFloat2(v1.texCoord);
	m_MISLight.t2 = ToFloat2(v2.texCoord);

	m_MISLight.emissionColor = ToFloat3(radiantExitance);

	/* TODO: LIGHT TEXTURE SAMPLING */
	m_MISLight.hasTexture = false;
	//m_MISLight.emissionTexture = ...;

	float3 ab = m_MISLight.p1 - m_MISLight.p0;
	float3 ac = m_MISLight.p2 - m_MISLight.p0;
	m_MISLight.area = 0.5f * length(cross(ab, ac));

	m_MISLight.power = m_MISLight.area * glm::length(radiantExitance);
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