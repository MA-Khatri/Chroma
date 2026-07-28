#include "network.hpp"

#include <plog/Log.h>

std::string to_json_string(std::string name, std::vector<int> arr) {
  nlohmann::json j;
  j[name] = arr;
  return j.dump();
}

ZMQ_Manager::ZMQ_Manager(std::string port) {
  try {
    _context = zmq::context_t(1);
    _request_socket = zmq::socket_t(_context, zmq::socket_type::req);

    _api_endpoint = "tcp://localhost:" + port;
    _request_socket.connect(_api_endpoint);

    _request_socket.set(zmq::sockopt::sndtimeo, DEFAULT_TIMEOUT);
    _request_socket.set(zmq::sockopt::rcvtimeo, DEFAULT_TIMEOUT);
  } catch (const std::exception &e) {
    PLOG_ERROR << "ZMQ Initialization error: " << e.what();
    return;
  }
}

ZMQ_Manager::~ZMQ_Manager() {
  _request_socket.close();
  _context.close();
}

std::string ZMQ_Manager::send_api_request(std::string request_string) {
  zmq::message_t request_message(request_string.size());
  memcpy(request_message.data(), request_string.data(), request_string.size());

  std::string reply = NACK_STRING;
  zmq::send_result_t status;

  _process_zmq_request(
      [&]() { status = _request_socket.send(request_message, zmq::send_flags::none); });

  if (status.has_value()) {
    zmq::message_t reply_message;

    std::string request_string;

    // Blocks until reply available
    PLOG_VERBOSE << "Waiting for reply";
    zmq::recv_result_t r;

    _process_zmq_request([&]() { r = _request_socket.recv(reply_message, zmq::recv_flags::none); });

    PLOG_VERBOSE << "Back from reply";

    if (r.has_value()) {
      reply = std::string(static_cast<char *>(reply_message.data()), reply_message.size());
    }

    if (reply == ACK_STRING) {
      PLOG_INFO << "Scanner API success: " << reply;
    } else {
      PLOG_ERROR << "Scanner API failure: " << reply;
    }
  } else {
    PLOG_ERROR << "Failed sending API request: " << request_string.substr(0, 80);
  }

  return reply;
}

void ZMQ_Manager::_process_zmq_request(const std::function<void()> &action) {
  try {
    action();
  } catch (const zmq::error_t &ex) {
    PLOG_ERROR << "ZMQ Exception: " << ex.what();
  } catch (const std::exception &ex) {
    PLOG_ERROR << "Exception: " << ex.what();
  }
}
