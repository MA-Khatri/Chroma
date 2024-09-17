#include "application.h"

int main()
{
	Application* app = new Application();

	/* Main Loop */
	while (!glfwWindowShouldClose(app->GetWindowHandle()))
	{
		app->RenderFrame();
	}

	return 0;
}