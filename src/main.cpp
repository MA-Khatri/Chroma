#include "application.hpp"
#include "raster_view.hpp"
#include "scene.hpp"

#include <memory>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

int main() {
  // Initialize logger with up to 3, 10 MB files (stored in build dir)
  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  static plog::RollingFileAppender<plog::TxtFormatter> fileAppender("chroma_log.txt",
                                                                    10 * 1024 * 1024, 3);
  plog::init(plog::debug, &fileAppender).addAppender(&consoleAppender);

  PLOG_INFO << "========== Starting Chroma ==========";

  // Initialize singleton app instance
  Application *app = Application::GetInstance();

  // Create Scenes
  PLOG_DEBUG << "Creating scenes...";
  app->PushScene(std::make_shared<Scene>(CreateScanningScene()));
  // app->PushScene(std::make_shared<Scene>(CreateTestScene()));
  PLOG_DEBUG << "Done creating scenes.";

  auto &scenes = app->GetAllScenes();
  if (scenes.size() > 0) {
    PLOG_DEBUG << "Setting active scene to \"" << scenes[0]->GetSceneName()
               << "\" (Scene ID: " << scenes[0]->m_SceneID << ")";
    app->SetActiveScene(scenes[0]->m_SceneID);
  } else {
    PLOG_ERROR << "No scenes available to set as active!";
  }

  // Create Layers
  PLOG_DEBUG << "Creating layers...";
  app->PushLayer(std::make_shared<RasterView>("Scanning"));
  app->PushLayer(std::make_shared<RasterView>("Model"));
  PLOG_DEBUG << "Done creating layers.";

  app->Run();

  PLOG_INFO << "========== Exiting Chroma ===========";

  return 0;
}