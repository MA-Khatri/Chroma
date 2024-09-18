#include "application.h"
#include "raster_view.h"
#include "raytrace_view.h"


int main()
{
	Application* app = new Application();

	app->PushLayer(std::make_shared<RasterView>());
	app->PushLayer(std::make_shared<RayTraceView>());

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

	app->Run();

	return 0;
}