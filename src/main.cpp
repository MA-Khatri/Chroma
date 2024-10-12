#define GLM_FORCE_CUDA /* Make sure this is defined before any include <glm/glm.hpp> */

#include "application.h"
#include "raster_view.h"
#include "raytrace_view.h"

int main()
{
	/* Initialize the application */
	Application* app = new Application();

	/* Make scenes */
	std::vector<std::shared_ptr<Scene>> scenes;
	scenes.push_back(std::make_shared<Scene>(Scene::DEFAULT));
	scenes.push_back(std::make_shared<Scene>(Scene::CORNELL_BOX));
	app->SetScenes(scenes);

	/* Create and initialize layers */
	app->PushLayer(std::make_shared<RasterView>());
	app->PushLayer(std::make_shared<RayTraceView>());


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