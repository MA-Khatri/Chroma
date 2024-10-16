#pragma once

#define GLM_ENABLE_EXPERIMENTAL

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtx/rotate_vector.hpp>
#include<glm/gtx/vector_angle.hpp>

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <string>
#include <map>

#include "common_enums.h"

class Camera
{
private:
	float m_NearPlane = 0.1f;
	float m_FarPlane = 1000.0f;

	/* Previous mouse position */
	glm::vec2 m_PrevMousePosn;


public:

	std::map<int, std::string> m_ControlModeNames = {
		{CONTROL_MODE_FREE_FLY, "Free Fly"},
		{CONTROL_MODE_ORBIT, "Orbit"},
	};

	std::map<int, std::string> m_ProjectionModeNames = {
		{PROJECTION_MODE_PERSPECTIVE, "Perspective"},
		{PROJECTION_MODE_ORTHOGRAPHIC, "Orthographic"},
		{PROJECTION_MODE_THIN_LENS, "Thin Lens"},
	};

	int m_ControlMode = CONTROL_MODE_FREE_FLY;
	int m_ProjectionMode = PROJECTION_MODE_PERSPECTIVE;

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

	/* The vertical field of view (in degrees) -- for perspective camera */
	float m_VFoV = 45.0f;

	/* FoV limits */
	float m_MinFoV = 5.0f; /* Minimum vertical FoV */
	float m_MaxFoV = 135.0f; /* Maximum vertical FoV */
	
	/* Scaling of the orthographic camera's view frustum w.r.t. the viewport size */
	float m_OrthoScale = 0.01f;
	float m_MinOrthoScale = 0.001f;

	/* Viewport bounds in pixels used to wrap the cursor when dragging */
	ImVec2 m_ViewportContentMin = ImVec2(0.0f, 0.0f);
	ImVec2 m_ViewportContentMax = ImVec2(0.0f, 0.0f);

	int m_Width; /* Viewport width (in pixels) */
	int m_Height; /* Viewport height (in pixels) */

	/* Is the left mouse key pressed? */
	bool m_LMB = false;

	/* Camera movement speed */
	float m_Speed = 0.01f;

	/* Camera rotation sensitivity */
	float m_Sensitivity = 100.0f;

	/* Variation angle of rays (in degrees) through each pixel used for thin lens. Larger = more blur. */
	float m_DefocusAngle = 1.0f;

	/* Distance from the camera origin to the plane of perfect focus (for thin lens) */
	float m_FocusDistance = 10.0f;

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

	/* Called on scrollwheel input */
	static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

	/* Check if current camera's properties match another camera's properties */
	bool IsCameraDifferent(Camera* camera);

}; /* class Camera */


