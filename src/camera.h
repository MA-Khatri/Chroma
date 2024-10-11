#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtx/rotate_vector.hpp>
#include<glm/gtx/vector_angle.hpp>

#include <GLFW/glfw3.h>

#include <imgui.h>

class Camera
{
private:
	float m_NearPlane;
	float m_FarPlane;

	/* Previous mouse position */
	glm::vec2 m_PrevMousePosn;


public:
	enum ControlMode
	{
		FREE_FLY,
		ORBIT
	};

	enum ProjectionMode
	{
		PERSPECTIVE,
		ORTHOGRAPHIC,
	};

	int m_ControlMode = FREE_FLY;
	int m_ProjectionMode = PERSPECTIVE;

	bool m_CameraUIUpdate = false; /* Set to true if camera values changed by UI */

	/* The position of the camera */
	glm::vec3 m_Position;

	/* The direction the camera is looking. Internally, the lookAt point is position + this orientation */
	glm::vec3 m_Orientation;

	/* The up vector(think orientation of the viewport) */
	glm::vec3 m_Up;

	/* For orbit camera, the point the camera orbits around */
	glm::vec3 m_OrbitOrigin = glm::vec3(0.0f);

	/* For orbit camera, the distance the camera is from the orbit origin */
	float m_OrbitDistance = 10.0f;

	/* For orbit camera, the angles */
	float m_OrbitTheta = 0.0f; /* Degrees around z-axis, from +x, (0, 360) */
	float m_OrbitPhi = 45.0f; /* Degrees from xy-plane, ~(-90, 90) */

	/* The camera matrix (initialized to identity matrix) -- will store the combined view-projection matrix */
	glm::mat4 m_Matrix = glm::mat4(1.0f);

	/* The matrix representing the camera's position and orientation */
	glm::mat4 m_ViewMatrix = glm::mat4(1.0f);

	/* The matrix representing the projection of the camera */
	glm::mat4 m_ProjectionMatrix = glm::mat4(1.0f);

	/* The vertical field of view (in degrees) */
	float m_VFoV;

	/* FoV limits */
	float m_MinFoV = 5.0f;
	float m_MaxFoV = 135.0f;

	/* Viewport bounds in pixels used to wrap the cursor when dragging */
	ImVec2 m_ViewportContentMin = ImVec2(0.0f, 0.0f);
	ImVec2 m_ViewportContentMax = ImVec2(0.0f, 0.0f);

	/* Width and Height of the viewport */
	int m_Width;
	int m_Height;

	/* Is the left mouse key pressed? */
	bool m_LMB = false;

	/* Camera movement speed */
	float m_Speed = 0.01f;

	/* Camera rotation sensitivity */
	float m_Sensitivity = 100.0f;

public:
	Camera(int width, int height, glm::vec3 position, glm::vec3 orientation, glm::vec3 up, float vfov = 45.0f, float near_plane = 0.1f, float far_plane = 1000.0f);
	Camera(); /* Constructs a camera with some default values */
	~Camera();

	/* Updates the view and projection matrices */
	void Update(float vFOVdeg, float nearPlane, float farPlane, int inWidth, int inHeight);

	/* Calculate the camera position and orientation for orbit mode */
	void UpdateOrbit();

	void UpdateViewMatrix();
	void UpdateProjectionMatrix(int width, int height);
	void UpdateProjectionMatrix(float vFOVdeg);
	void UpdateProjectionMatrix();

	/* Handles camera movement inputs. Returns boolean indicating if any inputs were recorded. */
	bool Inputs(GLFWwindow* window);

	/* Called on scrollwheel */
	static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

}; /* class Camera */


