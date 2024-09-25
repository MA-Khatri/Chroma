#include "camera.h"

Camera::Camera(int width, int height, glm::vec3 position, glm::vec3 orientation, glm::vec3 up, float vfov /* = 45.0f */, float near_plane /* = 0.1f */, float far_plane /* = 1000.0f */)
{
	Camera::m_Width = width;
	Camera::m_Height = height;
	Camera::position = position;
	Camera::orientation = orientation;
	Camera::up = up;
	Camera::vfov = vfov;

	Update(vfov, near_plane, far_plane, width, height);
}

Camera::~Camera()
{

}

void Camera::Update(float vFOVdeg, float nearPlane, float farPlane, int inWidth, int inHeight)
{
	vfov = vFOVdeg;

	m_Width = inWidth;
	m_Height = inHeight;

	/* Set the view and projection matrices using the lookAt and perspective glm functions */
	view_matrix = glm::lookAt(position, position + orientation, up);

	projection_matrix = glm::perspective(glm::radians(vFOVdeg), (float)(m_Width) / float(m_Height), nearPlane, farPlane);

	matrix = projection_matrix * view_matrix;
}

void Camera::UpdateViewMatrix()
{
	view_matrix = glm::lookAt(position, position + orientation, up);
	matrix = projection_matrix * view_matrix;
}

void Camera::Inputs(GLFWwindow* window)
{
	/* WASD keys for basic motion front/back, strafe left/right */ 
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		position += m_Speed * orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		position += m_Speed * -orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		position += m_Speed * -glm::normalize(glm::cross(orientation, up));
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		position += m_Speed * glm::normalize(glm::cross(orientation, up));
	}

	/* SPACE/CTRL moves up/down along up vector */
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		position += m_Speed * up;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
	{
		position += m_Speed * -up;
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
		/* 
		Later TODO: Make mouse wrap around viewport if dragged past the viewport bounds?
		*/

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

		/* Get new orientation for the camera */
		glm::vec3 newOrientation = glm::rotate(orientation, glm::radians(-rotX), glm::normalize(glm::cross(orientation, up)));

		/* Bound the up/down tilt between -5 to 5 radians */
		if (!(glm::angle(newOrientation, up) <= glm::radians(5.0f) or glm::angle(newOrientation, -up) <= glm::radians(5.0f)))
		{
			orientation = newOrientation;
		}

		/* Right/Left rotate (allowed to fully spin around) */
		orientation = glm::rotate(orientation, glm::radians(-rotY), up);

		/* Update prev mouse posn */
		m_PrevMousePosn = glm::vec2(mouseX, mouseY);
	}
	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
	{
		m_LMB = false;
	}

	/* Update the view matrix accounting for changed to camera position/orientation */
	UpdateViewMatrix();
}