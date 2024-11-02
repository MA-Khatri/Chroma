#include "camera.h"

#include <iostream>

#include "math_helpers.h"

Camera::Camera(int width, int height, glm::vec3 position, glm::vec3 orientation, glm::vec3 up, float vfov /* = 45.0f */, float near_plane /* = 0.1f */, float far_plane /* = 1000.0f */)
{
	m_Width = width;
	m_Height = height;
	m_Position = position;
	m_Orientation = orientation;
	m_Up = up;
	m_VFoV = vfov;
	m_NearPlane = near_plane;
	m_FarPlane = far_plane;

	Update(vfov, near_plane, far_plane, width, height);
}

Camera::Camera()
{
	m_Width = 100;
	m_Height = 100;
	m_Position = glm::vec3(17.0f, 0.0f, 5.0f);
	m_Orientation = glm::vec3(-1.0, 0.0, 0.0);
	m_Up = glm::vec3(0.0, 0.0, 1.0);
	m_VFoV = 45.0f;
	m_NearPlane = 0.1f;
	m_FarPlane = 1000.0f;

	Update(m_VFoV, m_NearPlane, m_FarPlane, m_Width, m_Height);
}

Camera::~Camera()
{

}

void Camera::Update(float vFOVdeg, float nearPlane, float farPlane, int inWidth, int inHeight)
{
	m_VFoV = vFOVdeg;

	m_Width = inWidth;
	m_Height = inHeight;
	m_NearPlane = nearPlane;
	m_FarPlane = farPlane;

	/* Set the view and projection matrices using the lookAt and perspective glm functions */
	m_ViewMatrix = glm::lookAt(m_Position, m_Position + m_Orientation, m_Up);

	UpdateProjectionMatrix();
}

void Camera::UpdateOrbit()
{
	/* Determine direction camera is looking */
	m_Orientation = -glm::normalize(glm::vec3(glm::cos(glm::radians(m_OrbitTheta)), glm::sin(glm::radians(m_OrbitTheta)), glm::tan(glm::radians(m_OrbitPhi))));

	/* Camera position is origin - orientation * distance */
	m_Position = m_OrbitOrigin - m_OrbitDistance * m_Orientation;
}

void Camera::UpdateViewMatrix()
{
	m_ViewMatrix = glm::lookAt(m_Position, m_Position + m_Orientation, m_Up);
	m_Matrix = m_ProjectionMatrix * m_ViewMatrix;
}

void Camera::UpdateProjectionMatrix(int width, int height)
{
	m_Width = width;
	m_Height = height;
	UpdateProjectionMatrix();
}

void Camera::UpdateProjectionMatrix(float vFOVdeg)
{
	m_VFoV = vFOVdeg;
	UpdateProjectionMatrix();
}

void Camera::UpdateProjectionMatrix()
{
	if (m_ProjectionMode == PROJECTION_MODE_ORTHOGRAPHIC)
	{
		float hw = 0.5f * m_OrthoScale * static_cast<float>(m_Width);
		float hh = 0.5f * m_OrthoScale * static_cast<float>(m_Height);
		m_ProjectionMatrix = glm::orthoRH_ZO(-hw, hw, -hh, hh, m_NearPlane, m_FarPlane);
	}
	else
	{
		m_ProjectionMatrix = glm::perspective(glm::radians(m_VFoV), static_cast<float>(m_Width) / static_cast<float>(m_Height), m_NearPlane, m_FarPlane);
	}

	m_Matrix = m_ProjectionMatrix * m_ViewMatrix;
}

bool Camera::Inputs(GLFWwindow* window)
{
	bool updated = false;

	if (m_ControlMode == CONTROL_MODE_FREE_FLY)
	{
		/* WASD keys for basic motion front/back, strafe left/right */
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		{
			m_Position += m_Speed * m_Orientation;
			updated = true;
		}
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		{
			m_Position += m_Speed * -m_Orientation;
			updated = true;
		}
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		{
			m_Position += m_Speed * -glm::normalize(glm::cross(m_Orientation, m_Up));
			updated = true;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		{
			m_Position += m_Speed * glm::normalize(glm::cross(m_Orientation, m_Up));
			updated = true;
		}

		/* SPACE/CTRL moves up/down along up vector */
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		{
			m_Position += m_Speed * m_Up;
			updated = true;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		{
			m_Position += m_Speed * -m_Up;
			updated = true;
		}

		/* Holding down shift increases m_Speed */
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		{
			m_Speed = 0.2f;
		}
		else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
		{
			m_Speed = 0.05f;
		}
	}

	/* Mouse drag for orientation */
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		/* Check if left mouse button was already pressed pressed */
		if (!m_LMB)
		{
			/* Update previous mouse position to current mouse position on first press */
			double mouseX;
			double mouseY;
			glfwGetCursorPos(window, &mouseX, &mouseY);
			m_PrevMousePosn = glm::vec2(mouseX, mouseY);

			m_LMB = true;
		}

		/* Get mouse position */
		double mouseX;
		double mouseY;
		glfwGetCursorPos(window, &mouseX, &mouseY);

		/* Mouse drag amounts for rotation */
		float yDrag = m_Sensitivity * static_cast<float>(mouseY - m_PrevMousePosn.y) / m_Height;
		float xDrag = m_Sensitivity * static_cast<float>(mouseX - m_PrevMousePosn.x) / m_Width;
		if (fabsf(yDrag) > 0.0f || fabsf(xDrag) > 0.0f) updated = true;

		if (m_ControlMode == CONTROL_MODE_FREE_FLY)
		{
			/* Get new orientation for the camera */
			glm::vec3 newOrientation = glm::rotate(m_Orientation, glm::radians(-yDrag), glm::normalize(glm::cross(m_Orientation, m_Up)));

			/* Bound the up/down tilt between -5 to 5 radians */
			if (!(glm::angle(newOrientation, m_Up) <= glm::radians(5.0f) or glm::angle(newOrientation, -m_Up) <= glm::radians(5.0f)))
			{
				m_Orientation = newOrientation;
			}

			/* Right/Left rotate (allowed to fully spin around) */
			m_Orientation = glm::rotate(m_Orientation, glm::radians(-xDrag), m_Up);
		}
		else if (m_ControlMode == CONTROL_MODE_ORBIT)
		{
			/* If control is pressed while clicking & dragging, vertical drag moves distance in/out */
			if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
			{
				m_OrbitDistance += yDrag;
				if (m_OrbitDistance < 0.01f) m_OrbitDistance = 0.01f;
			}
			/* If shift key is pressed while clicking & dragging, move the orbit origin along the plane orthogonal to the view direction */
			else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			{
				glm::vec3 uAxis = glm::normalize(glm::cross(m_Orientation, m_Up));
				glm::vec3 vAxis = m_Up;
				m_OrbitOrigin += 0.1f * (uAxis * -xDrag + vAxis * yDrag);
			}
			/* Otherwise, normal camera orbit */
			else
			{
				/* Keep theta in (0, 360) */
				m_OrbitTheta -= xDrag;
				if (m_OrbitTheta < 0.0f) m_OrbitTheta = 360.0f + fmodf(m_OrbitTheta, 360.0f);
				else m_OrbitTheta = fmodf(m_OrbitTheta, 360.0f);

				/* Keep phi in (-89, 89) */
				m_OrbitPhi += yDrag;
				if (m_OrbitPhi > 89.0f) m_OrbitPhi = 89.0f;
				if (m_OrbitPhi < -89.0f) m_OrbitPhi = -89.0f;
			}

			UpdateOrbit();
		}

		/* Wrap mouse around viewport if LMB is still pressed */
		int padding = 2; /* We add some padding to the viewport bc we're using ImGui::IsWindowHovered() to check for inputs */
		if (mouseX < m_ViewportContentMin[0] + padding)
		{
			mouseX = m_ViewportContentMax[0] - padding;
		}
		if (mouseX > m_ViewportContentMax[0] - padding)
		{
			mouseX = m_ViewportContentMin[0] + padding;
		}
		if (mouseY < m_ViewportContentMin[1]) /* No padding here to prevent issues when clicking to switch tabs */
		{
			mouseY = m_ViewportContentMax[1] - padding;
		}
		if (mouseY > m_ViewportContentMax[1] - padding)
		{
			mouseY = m_ViewportContentMin[1] + padding;
		}
		glfwSetCursorPos(window, (double)mouseX, (double)mouseY);

		/* Update prev mouse posn */
		m_PrevMousePosn = glm::vec2(mouseX, mouseY);
	}
	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
	{
		m_LMB = false;
	}

	/* Update the view matrix accounting for changed to camera position/orientation */
	UpdateViewMatrix();

	return updated;
}

void Camera::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	Camera* camera = (Camera*)glfwGetWindowUserPointer(window);

	if (camera->m_ProjectionMode == PROJECTION_MODE_ORTHOGRAPHIC)
	{
		camera->m_OrthoScale -= camera->m_MinOrthoScale * static_cast<float>(yoffset);
		if (camera->m_OrthoScale < camera->m_MinOrthoScale) camera->m_OrthoScale = camera->m_MinOrthoScale;
	}
	else
	{
		camera->m_VFoV -= static_cast<float>(yoffset);
		if (camera->m_VFoV < camera->m_MinFoV) camera->m_VFoV = camera->m_MinFoV;
		if (camera->m_VFoV > camera->m_MaxFoV) camera->m_VFoV = camera->m_MaxFoV;
	}

	/* Set this to true so that camera is updated */
	camera->m_CameraUIUpdate = true;
}


bool Camera::IsCameraDifferent(Camera* camera)
{
	Camera& c = *camera;
	if(!Close(m_Position, c.m_Position)) return true;
	if(!Close(m_Orientation, c.m_Orientation)) return true;
	if(!Close(m_Up, c.m_Up)) return true;
	if(!Close(m_OrbitOrigin, c.m_OrbitOrigin)) return true;
	if(!Close(m_OrbitDistance, c.m_OrbitDistance)) return true;
	if(!Close(m_OrbitTheta, c.m_OrbitTheta)) return true;
	if(!Close(m_OrbitPhi, c.m_OrbitPhi)) return true;
	if(!Close(m_VFoV, c.m_VFoV)) return true;
	if(!Close(m_OrthoScale, c.m_OrthoScale)) return true;
	if(!Close(static_cast<float>(m_Width), static_cast<float>(c.m_Width))) return true;
	if(!Close(static_cast<float>(m_Height), static_cast<float>(c.m_Height))) return true;

	return false;
}