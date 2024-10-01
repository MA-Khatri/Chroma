#include "application.h"
#include "raster_view.h"
#include "raytrace_view.h"

int main()
{
	/* Initialize the application */
	Application* app = new Application();


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