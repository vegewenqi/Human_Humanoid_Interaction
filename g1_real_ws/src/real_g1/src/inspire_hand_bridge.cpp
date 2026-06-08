// Bridge: ZED hand command (/hand_finger_angles) -> Unitree G1 Inspire hand.
//
// Input:
//   std_msgs/Float32MultiArray on /hand_finger_angles
//   Value convention: 1.0 = open, 0.0 = fist.
//
// Default ZED layout, input_layout=finger10:
//   [L thumb, L index, L middle, L ring, L pinky,
//    R thumb, R index, R middle, R ring, R pinky]
//
// Unitree Inspire command layout:
//   cmds[0..5]  = right hand
//   cmds[6..11] = left hand
//   per-hand order [pinky, ring, middle, index, thumb_bend, thumb_rotation]
//   q convention: 1.0 = open, 0.0 = close.
//
// Startup target is fist. After the first valid input, timeout or invalid input
// holds the last valid command.

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "unitree_go/msg/motor_cmds.hpp"
#include "unitree_go/msg/motor_states.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr size_t kPerHand = 6;
constexpr size_t kTotal = 12;
constexpr float kOpenQ = 1.0f;
constexpr float kFistQ = 0.0f;

enum class InputLayout
{
  Auto,
  Finger10,
  Finger12,
  Hand2,
};

float clamp01(float v)
{
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

bool valid_open_amount(float v)
{
  return std::isfinite(v) && v >= 0.0f && v <= 1.0f;
}

InputLayout parse_layout(const std::string & value)
{
  if (value == "auto") {
    return InputLayout::Auto;
  }
  if (value == "finger10") {
    return InputLayout::Finger10;
  }
  if (value == "finger12") {
    return InputLayout::Finger12;
  }
  if (value == "hand2") {
    return InputLayout::Hand2;
  }
  throw std::runtime_error("input_layout must be one of: auto, finger10, finger12, hand2");
}

std::string layout_name(InputLayout layout)
{
  switch (layout) {
    case InputLayout::Auto:
      return "auto";
    case InputLayout::Finger10:
      return "finger10";
    case InputLayout::Finger12:
      return "finger12";
    case InputLayout::Hand2:
      return "hand2";
  }
  return "unknown";
}

std::string format_q(const std::array<float, kTotal> & q)
{
  std::ostringstream oss;
  oss << "R[";
  for (size_t i = 0; i < kPerHand; ++i) {
    if (i > 0) {
      oss << " ";
    }
    oss << q[i];
  }
  oss << "] L[";
  for (size_t i = kPerHand; i < kTotal; ++i) {
    if (i > kPerHand) {
      oss << " ";
    }
    oss << q[i];
  }
  oss << "]";
  return oss.str();
}
}  // namespace

class InspireHandBridge : public rclcpp::Node
{
public:
  InspireHandBridge() : rclcpp::Node("inspire_hand_bridge")
  {
    declare_parameter<std::string>("finger_angles_topic", "/hand_finger_angles");
    declare_parameter<std::string>("command_topic", "/inspire/cmd");
    declare_parameter<std::string>("state_topic", "/inspire/state");
    declare_parameter<std::string>("controlled_side", "right");
    declare_parameter<std::string>("input_layout", "auto");
    declare_parameter<bool>("enable_motion", false);
    declare_parameter<bool>("publish_both_hands", true);
    declare_parameter<bool>("publish_state", false);
    declare_parameter<double>("command_timeout_sec", 0.5);
    declare_parameter<std::vector<double>>("hand_home_q", std::vector<double>(kTotal, kFistQ));
    declare_parameter<double>("shutdown_home_hold_sec", 0.5);
    declare_parameter<bool>("debug_log", false);
    declare_parameter<double>("debug_log_period_sec", 1.0);

    finger_angles_topic_ = get_parameter("finger_angles_topic").as_string();
    command_topic_ = get_parameter("command_topic").as_string();
    state_topic_ = get_parameter("state_topic").as_string();
    controlled_side_ = get_parameter("controlled_side").as_string();
    input_layout_name_ = get_parameter("input_layout").as_string();
    input_layout_ = parse_layout(input_layout_name_);
    enable_motion_ = get_parameter("enable_motion").as_bool();
    publish_both_hands_ = get_parameter("publish_both_hands").as_bool();
    publish_state_ = get_parameter("publish_state").as_bool();
    command_timeout_sec_ = get_parameter("command_timeout_sec").as_double();
    const auto hand_home_q_param = get_parameter("hand_home_q").as_double_array();
    shutdown_home_hold_sec_ = get_parameter("shutdown_home_hold_sec").as_double();
    debug_log_ = get_parameter("debug_log").as_bool();
    debug_log_period_sec_ = get_parameter("debug_log_period_sec").as_double();

    load_hand_home_q(hand_home_q_param);
    validate_parameters();

    q_ = hand_home_q_;
    last_cmd_time_ = now();

    auto dds_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
    command_pub_ = create_publisher<unitree_go::msg::MotorCmds>(command_topic_, dds_qos);

    auto input_qos = rclcpp::SensorDataQoS();
    finger_angles_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      finger_angles_topic_,
      input_qos,
      std::bind(&InspireHandBridge::on_finger_angles, this, std::placeholders::_1));

    if (publish_state_) {
      state_sub_ = create_subscription<unitree_go::msg::MotorStates>(
        state_topic_,
        dds_qos,
        std::bind(&InspireHandBridge::on_state, this, std::placeholders::_1));
    }

    timer_ = create_wall_timer(20ms, std::bind(&InspireHandBridge::on_timer, this));

    if (enable_motion_) {
      publish_command();
    }

    RCLCPP_INFO(
      get_logger(),
      "inspire_hand_bridge ready\n"
      "  input          : %s (std_msgs/Float32MultiArray)\n"
      "  output         : %s (unitree_go/MotorCmds)\n"
      "  state input    : %s%s\n"
      "  input_layout   : %s\n"
      "  controlled_side: %s\n"
      "  enable_motion  : %s\n"
      "  startup target : fist\n"
      "  shutdown target: hand_home_q\n"
      "  timeout policy : hold last valid command\n"
      "  value convention: /hand_finger_angles 1.0=open, 0.0=fist; q 1.0=open, 0.0=close",
      finger_angles_topic_.c_str(),
      command_topic_.c_str(),
      publish_state_ ? state_topic_.c_str() : "(disabled)",
      publish_state_ ? "" : "",
      input_layout_name_.c_str(),
      controlled_side_.c_str(),
      enable_motion_ ? "true" : "false");

    if (!publish_both_hands_) {
      RCLCPP_WARN(
        get_logger(),
        "publish_both_hands=false is accepted for launch compatibility, "
        "but this bridge still publishes 12 Inspire command entries.");
    }
    if (!enable_motion_) {
      RCLCPP_WARN(
        get_logger(),
        "enable_motion=false: bridge will parse /hand_finger_angles but will not publish motion commands.");
    }
  }

  void RequestSafeStop()
  {
    std::lock_guard<std::mutex> guard(shutdown_mtx_);
    if (shutdown_requested_) {
      return;
    }

    shutdown_requested_ = true;
    shutdown_done_ = false;
    shutdown_started_ = false;
    RCLCPP_WARN(get_logger(), "Ctrl-C detected. Sending hand_home_q before shutdown.");
  }

  bool SafeStopDone() const
  {
    std::lock_guard<std::mutex> guard(shutdown_mtx_);
    return shutdown_done_;
  }

private:
  void load_hand_home_q(const std::vector<double> & values)
  {
    if (values.size() != kTotal) {
      throw std::runtime_error("hand_home_q must have length 12");
    }

    for (size_t i = 0; i < kTotal; ++i) {
      const float value = static_cast<float>(values[i]);
      if (!valid_open_amount(value)) {
        throw std::runtime_error("hand_home_q entries must be finite values in [0, 1]");
      }
      hand_home_q_[i] = value;
    }
  }

  void validate_parameters()
  {
    if (
      controlled_side_ != "right" &&
      controlled_side_ != "left" &&
      controlled_side_ != "both")
    {
      throw std::runtime_error("controlled_side must be one of: right, left, both");
    }
    if (command_timeout_sec_ <= 0.0) {
      throw std::runtime_error("command_timeout_sec must be > 0");
    }
    if (shutdown_home_hold_sec_ < 0.0) {
      throw std::runtime_error("shutdown_home_hold_sec must be >= 0");
    }
    if (debug_log_period_sec_ <= 0.0) {
      throw std::runtime_error("debug_log_period_sec must be > 0");
    }
  }

  static void fill_unitree_hand(
    std::array<float, kTotal> & q,
    size_t base,
    float thumb,
    float index,
    float middle,
    float ring,
    float pinky)
  {
    q[base + 0] = clamp01(pinky);
    q[base + 1] = clamp01(ring);
    q[base + 2] = clamp01(middle);
    q[base + 3] = clamp01(index);
    q[base + 4] = clamp01(thumb);
    q[base + 5] = clamp01(thumb);
  }

  static void fill_unitree_hand12(
    std::array<float, kTotal> & q,
    size_t base,
    float thumb_bend,
    float thumb_rotation,
    float index,
    float middle,
    float ring,
    float pinky)
  {
    q[base + 0] = clamp01(pinky);
    q[base + 1] = clamp01(ring);
    q[base + 2] = clamp01(middle);
    q[base + 3] = clamp01(index);
    q[base + 4] = clamp01(thumb_bend);
    q[base + 5] = clamp01(thumb_rotation);
  }

  bool parse_finger10(
    const std::vector<float> & data,
    std::array<float, kTotal> & parsed_q)
  {
    if (data.size() != 10) {
      return false;
    }

    fill_unitree_hand(parsed_q, kPerHand, data[0], data[1], data[2], data[3], data[4]);
    fill_unitree_hand(parsed_q, 0, data[5], data[6], data[7], data[8], data[9]);
    return true;
  }

  bool parse_finger12(
    const std::vector<float> & data,
    std::array<float, kTotal> & parsed_q)
  {
    if (data.size() != 12) {
      return false;
    }

    fill_unitree_hand12(
      parsed_q,
      kPerHand,
      data[0],
      data[1],
      data[2],
      data[3],
      data[4],
      data[5]);
    fill_unitree_hand12(
      parsed_q,
      0,
      data[6],
      data[7],
      data[8],
      data[9],
      data[10],
      data[11]);
    return true;
  }

  bool parse_hand2(
    const std::vector<float> & data,
    std::array<float, kTotal> & parsed_q)
  {
    if (data.size() != 2) {
      return false;
    }

    const float left_open = data[0];
    const float right_open = data[1];
    fill_unitree_hand(parsed_q, kPerHand, left_open, left_open, left_open, left_open, left_open);
    fill_unitree_hand(parsed_q, 0, right_open, right_open, right_open, right_open, right_open);
    return true;
  }

  bool parse_input(
    const std::vector<float> & data,
    std::array<float, kTotal> & parsed_q,
    std::string & used_layout)
  {
    parsed_q.fill(kFistQ);
    for (const float value : data) {
      if (!valid_open_amount(value)) {
        return false;
      }
    }

    InputLayout layout = input_layout_;
    if (layout == InputLayout::Auto) {
      if (data.size() == 10) {
        layout = InputLayout::Finger10;
      } else if (data.size() == 12) {
        layout = InputLayout::Finger12;
      } else if (data.size() == 2) {
        layout = InputLayout::Hand2;
      }
    }

    used_layout = layout_name(layout);
    switch (layout) {
      case InputLayout::Finger10:
        return parse_finger10(data, parsed_q);
      case InputLayout::Finger12:
        return parse_finger12(data, parsed_q);
      case InputLayout::Hand2:
        return parse_hand2(data, parsed_q);
      case InputLayout::Auto:
        break;
    }
    return false;
  }

  void apply_controlled_side(const std::array<float, kTotal> & parsed_q)
  {
    if (controlled_side_ == "right" || controlled_side_ == "both") {
      for (size_t i = 0; i < kPerHand; ++i) {
        q_[i] = parsed_q[i];
      }
    }
    if (controlled_side_ == "left" || controlled_side_ == "both") {
      for (size_t i = kPerHand; i < kTotal; ++i) {
        q_[i] = parsed_q[i];
      }
    }
  }

  void on_finger_angles(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    if (ShutdownRequested()) {
      return;
    }

    std::array<float, kTotal> parsed_q{};
    std::string used_layout;
    if (!parse_input(msg->data, parsed_q, used_layout)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "could not parse /hand_finger_angles length=%zu with input_layout=%s",
        msg->data.size(),
        input_layout_name_.c_str());
      return;
    }

    apply_controlled_side(parsed_q);
    last_cmd_time_ = now();
    idle_ = false;

    if (enable_motion_) {
      publish_command();
    }

    if (debug_log_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        static_cast<int>(debug_log_period_sec_ * 1000.0),
        "input len=%zu layout=%s motion=%s %s",
        msg->data.size(),
        used_layout.c_str(),
        enable_motion_ ? "enabled" : "disabled",
        format_q(q_).c_str());
    }
  }

  void publish_command()
  {
    unitree_go::msg::MotorCmds out;
    out.cmds.resize(kTotal);
    for (size_t i = 0; i < kTotal; ++i) {
      out.cmds[i].q = q_[i];
    }
    command_pub_->publish(out);
  }

  void on_timer()
  {
    if (ShutdownRequested()) {
      run_shutdown_home();
      return;
    }

    const double dt = (now() - last_cmd_time_).seconds();
    if (dt <= command_timeout_sec_ || idle_) {
      return;
    }

    idle_ = true;
    RCLCPP_WARN(
      get_logger(),
      "no hand command for %.2fs (> %.2f); holding last valid command",
      dt,
      command_timeout_sec_);
  }

  void on_state(const unitree_go::msg::MotorStates::SharedPtr msg)
  {
    if (!debug_log_) {
      return;
    }
    RCLCPP_INFO_THROTTLE(
      get_logger(),
      *get_clock(),
      static_cast<int>(debug_log_period_sec_ * 1000.0),
      "state topic %s has %zu entries",
      state_topic_.c_str(),
      msg->states.size());
  }

  bool ShutdownRequested() const
  {
    std::lock_guard<std::mutex> guard(shutdown_mtx_);
    return shutdown_requested_;
  }

  void MarkSafeStopDone()
  {
    std::lock_guard<std::mutex> guard(shutdown_mtx_);
    shutdown_done_ = true;
    shutdown_requested_ = false;
  }

  void run_shutdown_home()
  {
    if (!shutdown_started_) {
      shutdown_started_ = true;
      shutdown_start_time_ = now();
      q_ = hand_home_q_;
      if (enable_motion_) {
        publish_command();
      }
    }

    q_ = hand_home_q_;
    if (enable_motion_) {
      publish_command();
    }

    if ((now() - shutdown_start_time_).seconds() >= shutdown_home_hold_sec_) {
      RCLCPP_WARN(get_logger(), "Hand shutdown home complete.");
      MarkSafeStopDone();
    }
  }

  std::string finger_angles_topic_;
  std::string command_topic_;
  std::string state_topic_;
  std::string controlled_side_;
  std::string input_layout_name_;
  InputLayout input_layout_{InputLayout::Auto};
  bool enable_motion_{false};
  bool publish_both_hands_{true};
  bool publish_state_{false};
  double command_timeout_sec_{0.5};
  double shutdown_home_hold_sec_{0.5};
  bool debug_log_{false};
  double debug_log_period_sec_{1.0};

  std::array<float, kTotal> q_{};
  std::array<float, kTotal> hand_home_q_{};
  rclcpp::Time last_cmd_time_;
  rclcpp::Time shutdown_start_time_;
  bool idle_{false};
  mutable std::mutex shutdown_mtx_;
  bool shutdown_requested_{false};
  bool shutdown_done_{false};
  bool shutdown_started_{false};

  rclcpp::Publisher<unitree_go::msg::MotorCmds>::SharedPtr command_pub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr finger_angles_sub_;
  rclcpp::Subscription<unitree_go::msg::MotorStates>::SharedPtr state_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::InitOptions init_options;
  init_options.shutdown_on_signal = false;
  rclcpp::init(argc, argv, init_options);

  static std::atomic<bool> g_shutdown_requested{false};
  std::signal(SIGINT, [](int) { g_shutdown_requested.store(true); });
  std::signal(SIGTERM, [](int) { g_shutdown_requested.store(true); });

  auto node = std::make_shared<InspireHandBridge>();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  bool safe_stop_requested = false;
  while (rclcpp::ok()) {
    executor.spin_some();

    if (g_shutdown_requested.load() && !safe_stop_requested) {
      node->RequestSafeStop();
      safe_stop_requested = true;
    }

    if (safe_stop_requested && node->SafeStopDone()) {
      break;
    }

    rclcpp::sleep_for(std::chrono::milliseconds(5));
  }

  executor.cancel();
  executor.remove_node(node);
  rclcpp::shutdown();
  return 0;
}
