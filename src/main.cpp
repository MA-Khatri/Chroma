#include "application.hpp"
#include "raster_view.hpp"
#include "raytrace_view.hpp"

#include <imgui.h>

#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

int main() {

  // Initialize logger with up to 3, 10 MB files (stored in build dir)
  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  static plog::RollingFileAppender<plog::TxtFormatter> fileAppender(
      "log.txt", 10 * 1024 * 1024, 3);
  plog::init(plog::debug, &fileAppender).addAppender(&consoleAppender);

  // Initialize singleton app instance
  Application *app = Application::GetInstance();

  // Create Layers
  app->PushLayer(std::make_shared<RasterView>());
  // app->PushLayer(std::make_shared<RayTraceView>());

  // Menubar setup
  app->SetMenubarCallback([app]() {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Exit")) {
        app->Close();
      }
      ImGui::EndMenu();
    }
  });

  app->Run();

  return 0;
}