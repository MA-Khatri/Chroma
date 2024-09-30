#include "application.h"
#include "raster_view.h"
#include "raytrace_view.h"

#include "optix.h"

void InitOptix()
{
	/* Check for Optix capable devices */
	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
	{
		std::cerr << "InitOptix(): no CUDA capable devices found!" << std::endl;
		exit(-1);
	}

	/* Initialize Optix */
	OPTIX_CHECK(optixInit());
}

int main()
{
	InitOptix();

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