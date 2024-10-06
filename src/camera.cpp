#include "camera.h"

Camera::Camera(int width, int height, glm::vec3 position, glm::vec3 orientation, glm::vec3 up, float vfov /* = 45.0f */, float near_plane /* = 0.1f */, float far_plane /* = 1000.0f */)
{
	Camera::m_Width = width;
	Camera::m_Height = height;
	Camera::position = position;
	Camera::orientation = orientation;
	Camera::up = up;
	Camera::vfov = vfov;
	Camera::m_NearPlane = near_plane;
	Camera::m_FarPlane = far_plane;

	Update(vfov, near_plane, far_plane, width, height);
}

Camera::Camera()
{
	Camera::m_Width = 100;
	Camera::m_Height = 100;
	Camera::position = glm::vec3(0.0f, 10.0f, 5.0f);
	Camera::orientation = glm::vec3(0.0, -1.0, 0.0);
	Camera::up = glm::vec3(0.0, 0.0, 1.0);
	Camera::vfov = 45.0f;
	Camera::m_NearPlane = 0.1f;
	Camera::m_FarPlane = 1000.0f;

	Update(vfov, m_NearPlane, m_FarPlane, m_Width, m_Height);
}

Camera::~Camera()
{

}

void Camera::Update(float vFOVdeg, float nearPlane, float farPlane, int inWidth, int inHeight)
{
	vfov = vFOVdeg;

	m_Width = inWidth;
	m_Height = inHeight;
	m_NearPlane = nearPlane;
	m_FarPlane = farPlane;

	/* Set the view and projection matrices using the lookAt and perspective glm functions */
	view_matrix = glm::lookAt(position, position + orientation, up);

	projection_matrix = glm::perspective(glm::radians(vfov), (float)(m_Width) / float(m_Height), m_NearPlane, m_FarPlane);

	matrix = projection_matrix * view_matrix;
}

void Camera::UpdateViewMatrix()
{
	view_matrix = glm::lookAt(position, position + orientation, up);
	matrix = projection_matrix * view_matrix;
}

void Camera::UpdateProjectionMatrix(int width, int height)
{
	m_Width = width;
	m_Height = height;

	projection_matrix = glm::perspective(glm::radians(vfov), (float)(m_Width) / float(m_Height), m_NearPlane, m_FarPlane);
	matrix = projection_matrix * view_matrix;
}

void Camera::UpdateProjectionMatrix(float vFOVdeg)
{
	vfov = vFOVdeg;

	projection_matrix = glm::perspective(glm::radians(vfov), (float)(m_Width) / float(m_Height), m_NearPlane, m_FarPlane);
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

		/* Wrap mouse around viewport if LMB is still pressed */
		int padding = 2; /* We add some padding to the viewport bc we're using ImGui::IsWindowHovered() to check for inputs */
		if (mouseX < viewportContentMin[0] + padding)
		{
			mouseX = viewportContentMax[0] - padding;
		}
		if (mouseX > viewportContentMax[0] - padding)
		{
			mouseX = viewportContentMin[0] + padding;
		}
		if (mouseY < viewportContentMin[1]) /* No padding here to prevent issues when clicking to switch tabs */
		{
			mouseY = viewportContentMax[1] - padding;
		}
		if (mouseY > viewportContentMax[1] - padding)
		{
			mouseY = viewportContentMin[1] + padding;
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
}