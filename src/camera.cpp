#include "camera.h"

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
	m_Position = glm::vec3(0.0f, 10.0f, 5.0f);
	m_Orientation = glm::vec3(0.0, -1.0, 0.0);
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

	m_ProjectionMatrix = glm::perspective(glm::radians(m_VFoV), (float)(m_Width) / float(m_Height), m_NearPlane, m_FarPlane);

	m_Matrix = m_ProjectionMatrix * m_ViewMatrix;
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
	m_ProjectionMatrix = glm::perspective(glm::radians(m_VFoV), (float)(m_Width) / float(m_Height), m_NearPlane, m_FarPlane);
	m_Matrix = m_ProjectionMatrix * m_ViewMatrix;
}

bool Camera::Inputs(GLFWwindow* window)
{
	bool updated = false;

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

	/* Mouse drag for orientation */
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		/* Check if left mouse button is pressed */
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
		float rotX = m_Sensitivity * (float)(mouseY - m_PrevMousePosn.y) / m_Height;
		float rotY = m_Sensitivity * (float)(mouseX - m_PrevMousePosn.x) / m_Width;
		if (fabsf(rotX) > 0.0f || fabsf(rotY) > 0.0f) updated = true;

		/* Get new orientation for the camera */
		glm::vec3 newOrientation = glm::rotate(m_Orientation, glm::radians(-rotX), glm::normalize(glm::cross(m_Orientation, m_Up)));

		/* Bound the up/down tilt between -5 to 5 radians */
		if (!(glm::angle(newOrientation, m_Up) <= glm::radians(5.0f) or glm::angle(newOrientation, -m_Up) <= glm::radians(5.0f)))
		{
			m_Orientation = newOrientation;
		}

		/* Right/Left rotate (allowed to fully spin around) */
		m_Orientation = glm::rotate(m_Orientation, glm::radians(-rotY), m_Up);

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