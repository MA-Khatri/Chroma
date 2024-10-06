#define GLM_FORCE_CUDA /* Make sure this is defined before any include <glm/glm.hpp> */

#include "application.h"
#include "raster_view.h"
#include "raytrace_view.h"

int main()
{
	/* Initialize the application */
	Application* app = new Application();

	std::shared_ptr<Scene> scene = std::make_shared<Scene>();

	/* Create and initialize layers */
	app->PushLayer(std::make_shared<RasterView>(scene));
	app->PushLayer(std::make_shared<RayTraceView>(scene));


	/* App menubar setup */
	app->SetMenubarCallback([app]()
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Exit"))
				{
					app->Close();
				}
				ImGui::EndMenu();
			}
		});


	/* Run the application */
	app->Run();


	return 0;
}