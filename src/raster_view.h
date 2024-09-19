#pragma once

#include "layer.h"
#include "image.h"

#include "camera.h"

class RasterView : public Layer {

	virtual void OnAttach(Application* app);


	virtual void OnDetach();

	virtual void OnUpdate();

	virtual void OnUIRender();

private:
	Application* m_AppHandle;
	GLFWwindow* m_WindowHandle;
	Camera* m_Camera;
	bool m_ViewportFocused = false;

	std::shared_ptr<Image> m_Image;
};
