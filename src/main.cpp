#include "application.hpp"
#include "raster_view.hpp"
#include "scanning_view.hpp"
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

  // Create Layers
  PLOG_DEBUG << "Creating layers...";
  app->PushLayer(std::make_shared<ScanningView>("Scanning"));
  app->PushLayer(std::make_shared<RasterView>("Model"));
  PLOG_DEBUG << "Done creating layers.";

  app->Run();

  PLOG_INFO << "========== Exiting Chroma ===========";

  return 0;
}