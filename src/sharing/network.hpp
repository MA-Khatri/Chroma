#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>

const static int DEFAULT_TIMEOUT = 3000; // milliseconds
const static std::string ACK_STRING = "ACK";
const static std::string NACK_STRING = "NACK";

std::string to_json_string(std::string name, std::vector<int> arr);

class ZMQ_Manager {
public:
  ZMQ_Manager(std::string port);
  ~ZMQ_Manager();

  std::string send_api_request(std::string request_string);

private:
  void _process_zmq_request(const std::function<void()> &action);

  zmq::context_t _context;
  zmq::socket_t _request_socket;
  std::string _api_endpoint;
};
