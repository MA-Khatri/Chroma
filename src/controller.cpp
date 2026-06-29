#include "controller.hpp"

#include <SDL3/SDL.h>
#include <plog/Log.h>

#include "application.hpp"
#include "imgui_impl_sdl3.h"

// Singleton instance
Controller *Controller::s_Instance = nullptr;

Controller::Controller() {
  // Menubar setup
  SetMenubarCallback([this]() {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Exit")) {
        PLOG_DEBUG << "Exit menu item clicked, closing application";
        this->Close();
      }
      ImGui::EndMenu();
    }
  });
}

void Controller::ProcessEvents() {
  int64_t deltaTime = Application::GetInstance()->GetTimestepNS();

  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    ImGui_ImplSDL3_ProcessEvent(&event);

    if (m_Camera) {
      m_Camera->Update(deltaTime, &event);
    }

    switch (event.type) {
    case SDL_EVENT_QUIT:
      PLOG_VERBOSE << "Received QUIT event";
      m_Running = false;
      break;

      // TODO: Handle other SDL events

    default:
      break;
    }
  }

  // Also call a per-frame camera update with no specific event so
  // continuous actions (held keys) are applied smoothly each frame.
  if (m_Camera) {
    m_Camera->Update(deltaTime, nullptr);
  }
}

void Controller::SetMenubarCallback(const std::function<void()> &menubarCallback) {
  m_MenubarCallback = menubarCallback;
}

std::function<void()> Controller::GetMenubarCallback() { return m_MenubarCallback; }