#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <csignal>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "g1/g1.hpp"

using namespace std::chrono_literals;

using LowCmd = unitree_hg::msg::LowCmd;
using LowState = unitree_hg::msg::LowState;
using JointState = sensor_msgs::msg::JointState;

constexpr double kPi = 3.14159265358979323846;
constexpr int G1_NUM_MOTOR = 29;

// Full 29-DoF URDF joint name order aligned with Unitree motor index
static const std::array<std::string, G1_NUM_MOTOR> kFullJointNames = {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
};

class G1ArmSdkBridge : public rclcpp::Node {
 public:
  static constexpr int NUM_ARM_JOINTS = 17;
  static constexpr auto NOT_USED_JOINT = G1Arm7JointIndex::NOT_USED_JOINT;

  G1ArmSdkBridge() : Node("g1_arm_sdk_bridge") {
    // ---------------- parameters ----------------
    this->declare_parameter<std::string>("qdes_topic", "/g1_upperbody_q_des_safe");
    this->declare_parameter<bool>("qdes_in_degrees", false);
    this->declare_parameter<std::string>("joint_state_topic", "/joint_states");

    this->declare_parameter<double>("control_dt", 0.02);
    this->declare_parameter<double>("ema_alpha", 0.10);
    this->declare_parameter<double>("max_joint_velocity", 0.20);
    this->declare_parameter<double>("topic_timeout_sec", 0.30);

    this->declare_parameter<double>("kp_arm", 60.0);
    this->declare_parameter<double>("kd_arm", 1.5);
    this->declare_parameter<double>("kp_waist", 40.0);
    this->declare_parameter<double>("kd_waist", 1.5);
    this->declare_parameter<double>("dq", 0.0);
    this->declare_parameter<double>("tau_ff", 0.0);

    this->declare_parameter<double>("weight_active", 1.0);
    this->declare_parameter<double>("weight_acquire_rate", 0.20);
    this->declare_parameter<double>("weight_release_rate", 0.30);
    this->declare_parameter<bool>("use_weight_ramp", true);

    this->declare_parameter<bool>("auto_move_to_home", true);
    this->declare_parameter<double>("home_transition_velocity", 0.06);
    this->declare_parameter<double>("home_hold_sec", 4.0);
    this->declare_parameter<double>("track_entry_blend_sec", 2.0);
    this->declare_parameter<double>("shutdown_return_velocity", 0.06);
    this->declare_parameter<double>("shutdown_hold_sec", 1.0);

    this->declare_parameter<bool>("hold_uncontrolled_joints_at_start_pose", true);

    // Input order:
    // [waist_roll, waist_pitch, l_sh_pitch, l_sh_roll, l_elbow, r_sh_pitch, r_sh_roll, r_elbow]
    this->declare_parameter<std::vector<double>>(
        "q_home_8", {0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708});
    this->declare_parameter<std::vector<double>>(
        "q_min_8", {-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472});
    this->declare_parameter<std::vector<double>>(
        "q_max_8", {0.52, 0.52, 2.6704, 2.2515, 2.0944, 2.6704, 1.5882, 2.0944});

    qdes_topic_ = this->get_parameter("qdes_topic").as_string();
    qdes_in_degrees_ = this->get_parameter("qdes_in_degrees").as_bool();
    joint_state_topic_ = this->get_parameter("joint_state_topic").as_string();

    control_dt_ = this->get_parameter("control_dt").as_double();
    ema_alpha_ = this->get_parameter("ema_alpha").as_double();
    max_joint_velocity_ = this->get_parameter("max_joint_velocity").as_double();
    topic_timeout_sec_ = this->get_parameter("topic_timeout_sec").as_double();

    kp_arm_ = this->get_parameter("kp_arm").as_double();
    kd_arm_ = this->get_parameter("kd_arm").as_double();
    kp_waist_ = this->get_parameter("kp_waist").as_double();
    kd_waist_ = this->get_parameter("kd_waist").as_double();
    dq_ = this->get_parameter("dq").as_double();
    tau_ff_ = this->get_parameter("tau_ff").as_double();

    weight_active_ = this->get_parameter("weight_active").as_double();
    weight_acquire_rate_ = this->get_parameter("weight_acquire_rate").as_double();
    weight_release_rate_ = this->get_parameter("weight_release_rate").as_double();
    use_weight_ramp_ = this->get_parameter("use_weight_ramp").as_bool();

    auto_move_to_home_ = this->get_parameter("auto_move_to_home").as_bool();
    home_transition_velocity_ = this->get_parameter("home_transition_velocity").as_double();
    home_hold_sec_ = this->get_parameter("home_hold_sec").as_double();
    track_entry_blend_sec_ = this->get_parameter("track_entry_blend_sec").as_double();
    shutdown_return_velocity_ = this->get_parameter("shutdown_return_velocity").as_double();
    shutdown_hold_sec_ = this->get_parameter("shutdown_hold_sec").as_double();

    hold_uncontrolled_joints_at_start_pose_ =
        this->get_parameter("hold_uncontrolled_joints_at_start_pose").as_bool();

    q_home_8_ = this->get_parameter("q_home_8").as_double_array();
    q_min_8_ = this->get_parameter("q_min_8").as_double_array();
    q_max_8_ = this->get_parameter("q_max_8").as_double_array();

    if (q_home_8_.size() != 8 || q_min_8_.size() != 8 || q_max_8_.size() != 8) {
      throw std::runtime_error("q_home_8 / q_min_8 / q_max_8 must all have length 8");
    }

    max_joint_delta_ = max_joint_velocity_ * control_dt_;
    home_max_joint_delta_ = home_transition_velocity_ * control_dt_;
    shutdown_max_joint_delta_ = shutdown_return_velocity_ * control_dt_;
    sleep_time_ = std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000.0));

    arm_joints_ = {
        G1Arm7JointIndex::LEFT_SHOULDER_PITCH,
        G1Arm7JointIndex::LEFT_SHOULDER_ROLL,
        G1Arm7JointIndex::LEFT_SHOULDER_YAW,
        G1Arm7JointIndex::LEFT_ELBOW,
        G1Arm7JointIndex::LEFT_WRIST_ROLL,
        G1Arm7JointIndex::LEFT_WRIST_PITCH,
        G1Arm7JointIndex::LEFT_WRIST_YAW,
        G1Arm7JointIndex::RIGHT_SHOULDER_PITCH,
        G1Arm7JointIndex::RIGHT_SHOULDER_ROLL,
        G1Arm7JointIndex::RIGHT_SHOULDER_YAW,
        G1Arm7JointIndex::RIGHT_ELBOW,
        G1Arm7JointIndex::RIGHT_WRIST_ROLL,
        G1Arm7JointIndex::RIGHT_WRIST_PITCH,
        G1Arm7JointIndex::RIGHT_WRIST_YAW,
        G1Arm7JointIndex::WAIST_YAW,
        G1Arm7JointIndex::WAIST_ROLL,
        G1Arm7JointIndex::WAIST_PITCH};

    current_jpos_des_.fill(0.0F);
    current_jpos_meas_.fill(0.0F);
    desired_17_.fill(0.0F);
    base_q_17_.fill(0.0F);

    latest_q_des_8_.assign(8, 0.0);
    q_target_safe_8_.assign(8, 0.0);
    q_home_start_8_.assign(8, 0.0);
    track_start_8_.assign(8, 0.0);

    has_qdes_ = false;
    has_lowstate_ = false;
    shutdown_released_ = false;
    safe_stop_requested_ = false;
    safe_stop_done_ = false;
    safe_return_initialized_ = false;
    safe_return_hold_started_ = false;
    home_initialized_ = false;
    home_reached_ = false;
    qdes_gate_open_ = false;
    weight_ = 0.0F;

    safe_return_target_17_.fill(0.0F);

    pub_arm_sdk_ = this->create_publisher<LowCmd>("/arm_sdk", 10);
    pub_joint_states_ = this->create_publisher<JointState>(joint_state_topic_, 10);

    sub_lowstate_ = this->create_subscription<LowState>(
        "/lowstate", 10,
        std::bind(&G1ArmSdkBridge::OnLowState, this, std::placeholders::_1));

    sub_qdes_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        qdes_topic_, 10,
        std::bind(&G1ArmSdkBridge::OnQdes, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(control_dt_),
        std::bind(&G1ArmSdkBridge::ControlLoop, this));

    last_qdes_time_ = this->now();
    last_log_time_ = this->now();
    home_transition_start_time_ = this->now();
    home_hold_start_time_ = this->now();
    track_entry_start_time_ = this->now();
    safe_return_hold_start_time_ = this->now();

    RCLCPP_INFO(this->get_logger(), "Pure ROS2 G1ArmSdkBridge started.");
    RCLCPP_INFO(this->get_logger(), "qdes_topic = %s", qdes_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "joint_state_topic = %s", joint_state_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "control_dt = %.4f", control_dt_);
    RCLCPP_INFO(this->get_logger(), "auto_move_to_home = %s", auto_move_to_home_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "hold_uncontrolled_joints_at_start_pose = %s",
                hold_uncontrolled_joints_at_start_pose_ ? "true" : "false");
  }

  void RequestSafeStop() {
    std::lock_guard<std::mutex> guard(shutdown_mtx_);
    if (safe_stop_requested_) {
      return;
    }
    safe_stop_requested_ = true;
    safe_stop_done_ = false;
    RCLCPP_WARN(this->get_logger(),
                "Ctrl-C detected. Entering SAFE_RETURN before shutdown.");
  }

  bool SafeStopDone() const {
    std::lock_guard<std::mutex> guard(shutdown_mtx_);
    return safe_stop_done_;
  }

  void RunSafeReturnStep(const rclcpp::Time &now) {
    if (!safe_return_initialized_) {
      current_jpos_des_ = current_jpos_meas_;
      desired_17_ = current_jpos_meas_;

      safe_return_target_17_ = base_q_17_;
      ApplyInput8ToDesired17(q_home_8_, base_q_17_, safe_return_target_17_);

      q_target_safe_8_ = q_home_8_;
      safe_return_initialized_ = true;
      safe_return_hold_started_ = false;

      RCLCPP_WARN(this->get_logger(),
                  "SAFE_RETURN started: moving to q_home before release.");
    }

    desired_17_ = safe_return_target_17_;
    StepTowardsDesired(shutdown_max_joint_delta_);

    if (use_weight_ramp_) {
      weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                       0.0F, static_cast<float>(weight_active_));
    } else {
      weight_ = static_cast<float>(weight_active_);
    }

    const double remain = GetMaxAbsError(current_jpos_des_, safe_return_target_17_);

    if (remain <= std::max(1e-4, shutdown_max_joint_delta_ * 0.5)) {
      if (!safe_return_hold_started_) {
        safe_return_hold_started_ = true;
        safe_return_hold_start_time_ = now;
        RCLCPP_WARN(this->get_logger(), "SAFE_RETURN reached q_home. Holding...");
      }

      const double hold_elapsed = (now - safe_return_hold_start_time_).seconds();
      if (hold_elapsed >= shutdown_hold_sec_) {
        if (use_weight_ramp_) {
          const float delta_w = static_cast<float>(weight_release_rate_ * control_dt_);
          weight_ = ClampF(weight_ - delta_w, 0.0F, static_cast<float>(weight_active_));
        } else {
          weight_ = 0.0F;
        }

        if (weight_ <= 1e-4F) {
          weight_ = 0.0F;
          {
            std::lock_guard<std::mutex> guard(shutdown_mtx_);
            safe_stop_done_ = true;
            safe_stop_requested_ = false;
            shutdown_released_ = true;
          }
          RCLCPP_WARN(this->get_logger(),
                      "SAFE_RETURN complete. Arm control released.");
        }
      }
    } else {
      safe_return_hold_started_ = false;
    }
  }

 private:
  enum class BridgeMode {
    HOLD_CURRENT,
    MOVE_TO_HOME,
    WAIT_FOR_QDES,
    TRACK_ENTRY,
    TRACK_QDES,
    SAFE_RETURN
  };

  static double Clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(x, hi));
  }

  static float ClampF(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
  }

  static double Lerp(double a, double b, double t) {
    return a + (b - a) * t;
  }

  void OnQdes(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    if (msg->data.size() != 8) {
      RCLCPP_WARN(this->get_logger(),
                  "Expected q_des dim=8, got %zu",
                  msg->data.size());
      return;
    }

    std::lock_guard<std::mutex> lock(mtx_);

    for (size_t i = 0; i < 8; ++i) {
      double v = static_cast<double>(msg->data[i]);
      if (qdes_in_degrees_) {
        v = v * kPi / 180.0;
      }
      latest_q_des_8_[i] = Clamp(v, q_min_8_[i], q_max_8_[i]);
    }

    has_qdes_ = true;
    last_qdes_time_ = this->now();
  }

  void OnLowState(const LowState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    last_state_ = *msg;

    for (size_t i = 0; i < arm_joints_.size(); ++i) {
      current_jpos_meas_[i] =
          last_state_.motor_state[static_cast<int>(arm_joints_[i])].q;
    }

    if (!has_lowstate_) {
      current_jpos_des_ = current_jpos_meas_;
      desired_17_ = current_jpos_meas_;
      base_q_17_ = current_jpos_meas_;

      q_home_start_8_ = ExtractInput8FromMeasured17(current_jpos_meas_);
      q_target_safe_8_ = q_home_start_8_;
      track_start_8_ = q_home_start_8_;

      home_transition_start_time_ = this->now();
      home_hold_start_time_ = this->now();
      track_entry_start_time_ = this->now();
      home_initialized_ = true;
      home_reached_ = !auto_move_to_home_;
      qdes_gate_open_ = false;
      has_lowstate_ = true;

      RCLCPP_INFO(this->get_logger(),
                  "Received first /lowstate. Bridge initialized from current measured pose.");
    }

    PublishJointStateFromLowState(*msg);
  }

  void PublishJointStateFromLowState(const LowState &msg) {
    JointState js;
    js.header.stamp = this->get_clock()->now();

    js.name.reserve(G1_NUM_MOTOR);
    js.position.reserve(G1_NUM_MOTOR);
    js.velocity.reserve(G1_NUM_MOTOR);
    js.effort.reserve(G1_NUM_MOTOR);

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      js.name.push_back(kFullJointNames[i]);
      js.position.push_back(static_cast<double>(msg.motor_state[i].q));
      js.velocity.push_back(static_cast<double>(msg.motor_state[i].dq));
      js.effort.push_back(static_cast<double>(msg.motor_state[i].tau_est));
    }

    pub_joint_states_->publish(js);
  }

  std::vector<double> ExtractInput8FromMeasured17(
      const std::array<float, NUM_ARM_JOINTS> &q17_meas) const {
    std::vector<double> q8(8, 0.0);
    q8[0] = static_cast<double>(q17_meas[15]);
    q8[1] = static_cast<double>(q17_meas[16]);
    q8[2] = static_cast<double>(q17_meas[0]);
    q8[3] = static_cast<double>(q17_meas[1]);
    q8[4] = static_cast<double>(q17_meas[3]);
    q8[5] = static_cast<double>(q17_meas[7]);
    q8[6] = static_cast<double>(q17_meas[8]);
    q8[7] = static_cast<double>(q17_meas[10]);

    for (size_t i = 0; i < 8; ++i) {
      q8[i] = Clamp(q8[i], q_min_8_[i], q_max_8_[i]);
    }
    return q8;
  }

  void ApplyInput8ToDesired17(const std::vector<double> &q8,
                              const std::array<float, NUM_ARM_JOINTS> &base_q17,
                              std::array<float, NUM_ARM_JOINTS> &q17) {
    q17 = base_q17;

    q17[0] = static_cast<float>(q8[2]);
    q17[1] = static_cast<float>(q8[3]);
    q17[3] = static_cast<float>(q8[4]);
    q17[7] = static_cast<float>(q8[5]);
    q17[8] = static_cast<float>(q8[6]);
    q17[10] = static_cast<float>(q8[7]);
    q17[15] = static_cast<float>(q8[0]);
    q17[16] = static_cast<float>(q8[1]);
  }

  void FillLowCmdFromCurrentDes(LowCmd &cmd) {
    for (size_t j = 0; j < arm_joints_.size(); ++j) {
      const int idx = static_cast<int>(arm_joints_[j]);
      const bool is_waist =
          (idx == static_cast<int>(G1Arm7JointIndex::WAIST_YAW) ||
           idx == static_cast<int>(G1Arm7JointIndex::WAIST_ROLL) ||
           idx == static_cast<int>(G1Arm7JointIndex::WAIST_PITCH));

      cmd.motor_cmd[idx].q = current_jpos_des_[j];
      cmd.motor_cmd[idx].dq = static_cast<float>(dq_);
      cmd.motor_cmd[idx].tau = static_cast<float>(tau_ff_);
      cmd.motor_cmd[idx].kp = is_waist ? static_cast<float>(kp_waist_)
                                       : static_cast<float>(kp_arm_);
      cmd.motor_cmd[idx].kd = is_waist ? static_cast<float>(kd_waist_)
                                       : static_cast<float>(kd_arm_);
    }

    cmd.motor_cmd[static_cast<int>(NOT_USED_JOINT)].q = weight_;
  }

  BridgeMode GetCurrentMode(bool qdes_fresh) {
    {
      std::lock_guard<std::mutex> guard(shutdown_mtx_);
      if (safe_stop_requested_) {
        return BridgeMode::SAFE_RETURN;
      }
    }

    if (!home_initialized_) {
      return BridgeMode::HOLD_CURRENT;
    }
    if (!home_reached_) {
      return BridgeMode::MOVE_TO_HOME;
    }
    if (!qdes_gate_open_) {
      qdes_gate_open_ = true;
      track_entry_start_time_ = this->now();
      track_start_8_ = q_home_8_;
      q_target_safe_8_ = q_home_8_;
      RCLCPP_INFO(this->get_logger(), "Initialization phase complete. q_des gate is now open.");
      return BridgeMode::WAIT_FOR_QDES;
    }
    if (qdes_fresh && !track_entry_done_) {
      return BridgeMode::TRACK_ENTRY;
    }
    if (qdes_fresh) {
      return BridgeMode::TRACK_QDES;
    }
    track_entry_done_ = false;
    return BridgeMode::WAIT_FOR_QDES;
  }

  std::vector<double> ComputeHomeReference(const rclcpp::Time &now) {
    if (!auto_move_to_home_) {
      home_reached_ = true;
      return q_home_8_;
    }

    const double elapsed = (now - home_transition_start_time_).seconds();
    const double move_duration = GetHomeMoveDuration();

    std::vector<double> q8(8, 0.0);
    for (size_t i = 0; i < 8; ++i) {
      const double dist = std::fabs(q_home_8_[i] - q_home_start_8_[i]);
      const double duration =
          (home_transition_velocity_ > 1e-6) ? dist / home_transition_velocity_ : 0.0;
      const double ti = (duration > 1e-6) ? Clamp(elapsed / duration, 0.0, 1.0) : 1.0;
      q8[i] = Lerp(q_home_start_8_[i], q_home_8_[i], ti);
    }

    if (elapsed >= move_duration) {
      if ((now - home_hold_start_time_).seconds() < control_dt_ * 1.5) {
        RCLCPP_INFO(this->get_logger(), "Reached q_home. Holding before enabling q_des tracking...");
      }
      if ((now - home_transition_start_time_).seconds() >= move_duration + home_hold_sec_) {
        home_reached_ = true;
        q_target_safe_8_ = q_home_8_;
        track_start_8_ = q_home_8_;
        track_entry_done_ = false;
        RCLCPP_INFO(this->get_logger(), "Home hold complete.");
      }
    } else {
      home_hold_start_time_ = now;
    }

    return q8;
  }

  std::vector<double> ComputeTrackEntryReference(const rclcpp::Time &now, bool qdes_fresh) {
    if (!qdes_fresh) {
      track_entry_done_ = false;
      return q_home_8_;
    }

    const double elapsed = (now - track_entry_start_time_).seconds();
    const double duration = std::max(control_dt_, track_entry_blend_sec_);
    const double t = Clamp(elapsed / duration, 0.0, 1.0);

    std::vector<double> target8 = latest_q_des_8_;
    for (size_t i = 0; i < 8; ++i) {
      target8[i] = Clamp(target8[i], q_min_8_[i], q_max_8_[i]);
      target8[i] = ema_alpha_ * target8[i] + (1.0 - ema_alpha_) * q_target_safe_8_[i];
    }

    std::vector<double> q8(8, 0.0);
    for (size_t i = 0; i < 8; ++i) {
      q8[i] = Lerp(track_start_8_[i], target8[i], t);
    }

    if (t >= 1.0) {
      track_entry_done_ = true;
    }

    return q8;
  }

  double GetHomeMoveDuration() const {
    double max_duration = 0.0;
    for (size_t i = 0; i < 8; ++i) {
      const double dist = std::fabs(q_home_8_[i] - q_home_start_8_[i]);
      const double duration =
          (home_transition_velocity_ > 1e-6) ? dist / home_transition_velocity_ : 0.0;
      max_duration = std::max(max_duration, duration);
    }
    return max_duration;
  }

  static double GetMaxAbsError(const std::array<float, NUM_ARM_JOINTS> &a,
                               const std::array<float, NUM_ARM_JOINTS> &b) {
    double v = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      v = std::max(v, std::fabs(static_cast<double>(a[i] - b[i])));
    }
    return v;
  }

  void StepTowardsDesired(double max_delta) {
    for (size_t j = 0; j < current_jpos_des_.size(); ++j) {
      float err = desired_17_[j] - current_jpos_des_[j];
      err = ClampF(err, -static_cast<float>(max_delta), static_cast<float>(max_delta));
      current_jpos_des_[j] += err;
    }
  }

  void ControlLoop() {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!has_lowstate_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "Waiting for /lowstate...");
      return;
    }

    const rclcpp::Time now = this->now();
    const bool qdes_fresh =
        has_qdes_ && ((now - last_qdes_time_).seconds() <= topic_timeout_sec_);

    const BridgeMode mode = GetCurrentMode(qdes_fresh);

    double used_max_delta = max_joint_delta_;
    std::vector<double> q_ref_8 = q_target_safe_8_;

    if (mode == BridgeMode::SAFE_RETURN) {
      RunSafeReturnStep(now);
    } else if (mode == BridgeMode::HOLD_CURRENT) {
      desired_17_ = current_jpos_meas_;
      weight_ = 0.0F;
      StepTowardsDesired(used_max_delta);
    } else if (mode == BridgeMode::MOVE_TO_HOME) {
      q_ref_8 = ComputeHomeReference(now);
      q_target_safe_8_ = q_ref_8;
      used_max_delta = home_max_joint_delta_;
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput8ToDesired17(q_ref_8, base_q_17_, desired_17_);
      StepTowardsDesired(used_max_delta);
    } else if (mode == BridgeMode::WAIT_FOR_QDES) {
      q_ref_8 = q_home_8_;
      q_target_safe_8_ = q_ref_8;
      track_start_8_ = q_home_8_;
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput8ToDesired17(q_ref_8, base_q_17_, desired_17_);
      StepTowardsDesired(used_max_delta);
    } else if (mode == BridgeMode::TRACK_ENTRY) {
      used_max_delta = max_joint_delta_;
      q_ref_8 = ComputeTrackEntryReference(now, qdes_fresh);
      q_target_safe_8_ = q_ref_8;
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput8ToDesired17(q_target_safe_8_, base_q_17_, desired_17_);
      StepTowardsDesired(used_max_delta);
    } else {
      q_ref_8 = latest_q_des_8_;
      for (size_t i = 0; i < 8; ++i) {
        q_ref_8[i] = Clamp(q_ref_8[i], q_min_8_[i], q_max_8_[i]);
        q_target_safe_8_[i] =
            ema_alpha_ * q_ref_8[i] + (1.0 - ema_alpha_) * q_target_safe_8_[i];
      }
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput8ToDesired17(q_target_safe_8_, base_q_17_, desired_17_);
      StepTowardsDesired(used_max_delta);
    }

    LowCmd cmd;
    FillLowCmdFromCurrentDes(cmd);
    pub_arm_sdk_->publish(cmd);

    if ((now - last_log_time_).seconds() > 1.0) {
      auto deg = [](double rad) { return rad * 180.0 / kPi; };
      const char *mode_str = "unknown";
      switch (mode) {
        case BridgeMode::HOLD_CURRENT:
          mode_str = "hold_current";
          break;
        case BridgeMode::MOVE_TO_HOME:
          mode_str = "move_to_home";
          break;
        case BridgeMode::WAIT_FOR_QDES:
          mode_str = "wait_for_qdes";
          break;
        case BridgeMode::TRACK_ENTRY:
          mode_str = "track_entry";
          break;
        case BridgeMode::TRACK_QDES:
          mode_str = "track_qdes";
          break;
        case BridgeMode::SAFE_RETURN:
          mode_str = "safe_return";
          break;
      }
      RCLCPP_INFO(
          this->get_logger(),
          "[bridge] mode=%s gate=%s weight=%.2f fresh=%s | q8_deg: wr=%.1f wp=%.1f lsp=%.1f lsr=%.1f le=%.1f rsp=%.1f rsr=%.1f re=%.1f",
          mode_str,
          qdes_gate_open_ ? "open" : "closed",
          weight_,
          qdes_fresh ? "true" : "false",
          deg(q_target_safe_8_[0]),
          deg(q_target_safe_8_[1]),
          deg(q_target_safe_8_[2]),
          deg(q_target_safe_8_[3]),
          deg(q_target_safe_8_[4]),
          deg(q_target_safe_8_[5]),
          deg(q_target_safe_8_[6]),
          deg(q_target_safe_8_[7]));
      last_log_time_ = now;
    }
  }

 private:
  rclcpp::Publisher<LowCmd>::SharedPtr pub_arm_sdk_;
  rclcpp::Publisher<JointState>::SharedPtr pub_joint_states_;
  rclcpp::Subscription<LowState>::SharedPtr sub_lowstate_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_qdes_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mtx_;
  mutable std::mutex shutdown_mtx_;

  std::string qdes_topic_;
  std::string joint_state_topic_;
  bool qdes_in_degrees_;
  bool use_weight_ramp_;
  bool hold_uncontrolled_joints_at_start_pose_;

  double control_dt_;
  double ema_alpha_;
  double max_joint_velocity_;
  double max_joint_delta_;
  double topic_timeout_sec_;
  double home_transition_velocity_;
  double home_max_joint_delta_;
  double track_entry_blend_sec_;
  double shutdown_return_velocity_;
  double shutdown_max_joint_delta_;
  double shutdown_hold_sec_;

  double kp_arm_;
  double kd_arm_;
  double kp_waist_;
  double kd_waist_;
  double dq_;
  double tau_ff_;

  double weight_active_;
  double weight_acquire_rate_;
  double weight_release_rate_;

  bool auto_move_to_home_;
  double home_hold_sec_;

  std::vector<double> q_home_8_;
  std::vector<double> q_min_8_;
  std::vector<double> q_max_8_;

  std::chrono::milliseconds sleep_time_{};

  LowState last_state_;
  std::array<G1Arm7JointIndex, NUM_ARM_JOINTS> arm_joints_;

  std::array<float, NUM_ARM_JOINTS> current_jpos_des_;
  std::array<float, NUM_ARM_JOINTS> current_jpos_meas_;
  std::array<float, NUM_ARM_JOINTS> desired_17_;
  std::array<float, NUM_ARM_JOINTS> base_q_17_;

  std::vector<double> latest_q_des_8_;
  std::vector<double> q_target_safe_8_;
  std::vector<double> q_home_start_8_;
  std::vector<double> track_start_8_;

  bool has_qdes_;
  bool has_lowstate_;
  bool shutdown_released_;
  bool safe_stop_requested_{false};
  bool safe_stop_done_{false};
  bool safe_return_initialized_{false};
  bool safe_return_hold_started_{false};
  bool home_initialized_;
  bool home_reached_;
  bool qdes_gate_open_;
  bool track_entry_done_{false};
  float weight_;

  std::array<float, NUM_ARM_JOINTS> safe_return_target_17_{};
  rclcpp::Time safe_return_hold_start_time_;

  rclcpp::Time last_qdes_time_;
  rclcpp::Time last_log_time_;
  rclcpp::Time home_transition_start_time_;
  rclcpp::Time home_hold_start_time_;
  rclcpp::Time track_entry_start_time_;
};

int main(int argc, char **argv) {
  rclcpp::InitOptions init_options;
  init_options.shutdown_on_signal = false;
  rclcpp::init(argc, argv, init_options);

  static std::atomic<bool> g_sigint_requested{false};
  std::signal(SIGINT, [](int) {
    g_sigint_requested.store(true);
  });

  auto node = std::make_shared<G1ArmSdkBridge>();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  bool safe_stop_requested = false;

  while (rclcpp::ok()) {
    executor.spin_some();

    if (g_sigint_requested.load() && !safe_stop_requested) {
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
