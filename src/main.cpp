#include "application.hpp"

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

  app->Run();

  return 0;
}