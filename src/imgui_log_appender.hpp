#pragma once
#include <deque>
#include <imgui.h>
#include <mutex>
#include <plog/Appenders/IAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Log.h>
#include <string>


class ImGuiLogAppender : public plog::IAppender {
public:
  struct LogEntry {
    plog::Severity severity;
    std::string message;
  };

  // Called by plog from whatever thread logs - keep it cheap and thread-safe
  virtual void write(const plog::Record &record) override {
    plog::util::nstring formatted = plog::TxtFormatter::format(record);

    // TxtFormatter appends a trailing newline; strip it since ImGui draws per-line
    std::string msg(formatted.begin(), formatted.end());
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r'))
      msg.pop_back();

    std::lock_guard<std::mutex> lock(m_mutex);
    m_entries.push_back({record.getSeverity(), std::move(msg)});
    if (m_entries.size() > m_maxEntries)
      m_entries.pop_front();
  }

  void draw(const char *title = "Log", bool *open = nullptr) {
    ImGui::Begin(title, open);

    static ImGuiTextFilter filter;
    filter.Draw("Filter", 180);
    ImGui::SameLine();
    bool doClear = ImGui::Button("Clear");
    ImGui::SameLine();
    bool doCopy = ImGui::Button("Copy");
    ImGui::SameLine();
    ImGui::Checkbox("Auto-scroll", &m_autoScroll);
    ImGui::Separator();

    ImGui::BeginChild("scroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    std::lock_guard<std::mutex> lock(m_mutex);

    if (doClear)
      m_entries.clear();
    if (doCopy)
      ImGui::LogToClipboard();

    // For very high log volume, swap this loop for ImGuiListClipper
    for (auto &entry : m_entries) {
      if (!filter.PassFilter(entry.message.c_str()))
        continue;

      ImVec4 color;
      bool hasColor = true;
      switch (entry.severity) {
      case plog::fatal:
        color = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
        break;
      case plog::error:
        color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
        break;
      case plog::warning:
        color = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);
        break;
      case plog::info:
        color = ImVec4(0.6f, 0.9f, 1.0f, 1.0f);
        break;
      case plog::debug:
        color = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
        break;
      default:
        hasColor = false;
        break;
      }

      if (hasColor)
        ImGui::PushStyleColor(ImGuiCol_Text, color);
      ImGui::TextUnformatted(entry.message.c_str());
      if (hasColor)
        ImGui::PopStyleColor();
    }

    if (m_autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
      ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
    ImGui::End();
  }

private:
  std::mutex m_mutex;
  std::deque<LogEntry> m_entries;
  size_t m_maxEntries = 2000;
  bool m_autoScroll = true;
};