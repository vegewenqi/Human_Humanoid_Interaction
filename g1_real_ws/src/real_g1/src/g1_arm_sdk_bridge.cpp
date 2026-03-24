#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "g1/g1.hpp"

using namespace std::chrono_literals;

using LowCmd = unitree_hg::msg::LowCmd;
using LowState = unitree_hg::msg::LowState;

constexpr double kPi = 3.14159265358979323846;

class G1ArmSdkBridge : public rclcpp::Node {
 public:
  static constexpr int NUM_ARM_JOINTS = 17;
  static constexpr auto NOT_USED_JOINT = G1Arm7JointIndex::NOT_USED_JOINT;

  G1ArmSdkBridge() : Node("g1_arm_sdk_bridge") {
    // ---------------- parameters ----------------
    this->declare_parameter<std::string>("qdes_topic", "/g1_upperbody_q_des");
    this->declare_parameter<bool>("qdes_in_degrees", false);

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
    this->declare_parameter<double>("weight_release_rate", 0.15);
    this->declare_parameter<double>("shutdown_release_sec", 2.0);
    this->declare_parameter<bool>("use_weight_ramp", true);

    this->declare_parameter<bool>("auto_move_to_home", true);
    this->declare_parameter<double>("home_transition_velocity", 0.06);
    this->declare_parameter<double>("home_hold_sec", 4.0);
    this->declare_parameter<double>("track_entry_blend_sec", 2.0);
    this->declare_parameter<double>("shutdown_return_velocity", 0.06);
    this->declare_parameter<double>("shutdown_hold_sec", 1.0);

    this->declare_parameter<bool>("hold_uncontrolled_joints_at_start_pose", true);

    // Input order:
    // [waist_roll, waist_pitch, l_sh_roll, l_elbow, r_sh_roll, r_elbow]
    this->declare_parameter<std::vector<double>>(
        "q_home_6", {0.0, 0.0, 0.0, 1.5708, 0.0, 1.5708});
    this->declare_parameter<std::vector<double>>(
        "q_min_6", {-0.52, -0.52, 0.0, -1.0472, -2.2515, -1.0472});
    this->declare_parameter<std::vector<double>>(
        "q_max_6", {0.52, 0.52, 2.2515, 2.0944, 0.0, 2.0944});

    qdes_topic_ = this->get_parameter("qdes_topic").as_string();
    qdes_in_degrees_ = this->get_parameter("qdes_in_degrees").as_bool();

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
    shutdown_release_sec_ = this->get_parameter("shutdown_release_sec").as_double();
    use_weight_ramp_ = this->get_parameter("use_weight_ramp").as_bool();

    auto_move_to_home_ = this->get_parameter("auto_move_to_home").as_bool();
    home_transition_velocity_ = this->get_parameter("home_transition_velocity").as_double();
    home_hold_sec_ = this->get_parameter("home_hold_sec").as_double();
    track_entry_blend_sec_ = this->get_parameter("track_entry_blend_sec").as_double();
    shutdown_return_velocity_ = this->get_parameter("shutdown_return_velocity").as_double();
    shutdown_hold_sec_ = this->get_parameter("shutdown_hold_sec").as_double();

    hold_uncontrolled_joints_at_start_pose_ =
        this->get_parameter("hold_uncontrolled_joints_at_start_pose").as_bool();

    q_home_6_ = this->get_parameter("q_home_6").as_double_array();
    q_min_6_ = this->get_parameter("q_min_6").as_double_array();
    q_max_6_ = this->get_parameter("q_max_6").as_double_array();

    if (q_home_6_.size() != 6 || q_min_6_.size() != 6 || q_max_6_.size() != 6) {
      throw std::runtime_error("q_home_6 / q_min_6 / q_max_6 must all have length 6");
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

    latest_q_des_6_.assign(6, 0.0);
    q_target_safe_6_.assign(6, 0.0);
    q_home_start_6_.assign(6, 0.0);
    track_start_6_.assign(6, 0.0);

    has_qdes_ = false;
    has_lowstate_ = false;
    shutdown_released_ = false;
    home_initialized_ = false;
    home_reached_ = false;
    qdes_gate_open_ = false;
    weight_ = 0.0F;

    pub_arm_sdk_ = this->create_publisher<LowCmd>("/arm_sdk", 10);

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

    RCLCPP_INFO(this->get_logger(), "Pure ROS2 G1ArmSdkBridge started.");
    RCLCPP_INFO(this->get_logger(), "qdes_topic = %s", qdes_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "control_dt = %.4f", control_dt_);
    RCLCPP_INFO(this->get_logger(), "auto_move_to_home = %s", auto_move_to_home_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "hold_uncontrolled_joints_at_start_pose = %s",
                hold_uncontrolled_joints_at_start_pose_ ? "true" : "false");
  }

  void ReleaseControlSafely() {
    std::lock_guard<std::mutex> guard(shutdown_mtx_);
    if (shutdown_released_) {
      return;
    }
    shutdown_released_ = true;

    if (!has_lowstate_) {
      return;
    }

    RCLCPP_WARN(this->get_logger(), "Shutdown detected. Returning smoothly to q_home before release...");

    if (timer_) {
      timer_->cancel();
    }

    std::array<float, NUM_ARM_JOINTS> shutdown_target_17 = base_q_17_;
    ApplyInput6ToDesired17(q_home_6_, base_q_17_, shutdown_target_17);

    const int move_steps = std::max(
        1, static_cast<int>(std::ceil(GetMaxAbsError(current_jpos_des_, shutdown_target_17) /
                                      std::max(1e-6, shutdown_max_joint_delta_))));

    for (int i = 0; i < move_steps; ++i) {
      {
        std::lock_guard<std::mutex> lock(mtx_);
        desired_17_ = shutdown_target_17;
        StepTowardsDesired(shutdown_max_joint_delta_);
        if (use_weight_ramp_) {
          weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                           0.0F, static_cast<float>(weight_active_));
        } else {
          weight_ = static_cast<float>(weight_active_);
        }

        LowCmd cmd;
        FillLowCmdFromCurrentDes(cmd);
        pub_arm_sdk_->publish(cmd);
      }
      rclcpp::sleep_for(sleep_time_);
    }

    const int hold_steps = std::max(1, static_cast<int>(std::round(shutdown_hold_sec_ / control_dt_)));
    for (int i = 0; i < hold_steps; ++i) {
      {
        std::lock_guard<std::mutex> lock(mtx_);
        desired_17_ = shutdown_target_17;
        StepTowardsDesired(shutdown_max_joint_delta_);

        LowCmd cmd;
        FillLowCmdFromCurrentDes(cmd);
        pub_arm_sdk_->publish(cmd);
      }
      rclcpp::sleep_for(sleep_time_);
    }

    const int release_steps =
        std::max(1, static_cast<int>(std::round(shutdown_release_sec_ / control_dt_)));
    for (int i = 0; i < release_steps; ++i) {
      {
        std::lock_guard<std::mutex> lock(mtx_);
        if (use_weight_ramp_) {
          const float delta_w = static_cast<float>(weight_release_rate_ * control_dt_);
          weight_ = ClampF(weight_ - delta_w, 0.0F, static_cast<float>(weight_active_));
        } else {
          weight_ = 0.0F;
        }

        LowCmd cmd;
        FillLowCmdFromCurrentDes(cmd);
        pub_arm_sdk_->publish(cmd);
      }
      rclcpp::sleep_for(sleep_time_);
    }

    {
      std::lock_guard<std::mutex> lock(mtx_);
      weight_ = 0.0F;
      LowCmd cmd;
      FillLowCmdFromCurrentDes(cmd);
      pub_arm_sdk_->publish(cmd);
    }

    RCLCPP_WARN(this->get_logger(), "Returned to q_home and released arm control safely.");
  }

 private:
  enum class BridgeMode {
    HOLD_CURRENT,
    MOVE_TO_HOME,
    WAIT_FOR_QDES,
    TRACK_ENTRY,
    TRACK_QDES
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
    if (msg->data.size() != 6) {
      RCLCPP_WARN(this->get_logger(),
                  "Expected /g1_upperbody_q_des dim=6, got %zu",
                  msg->data.size());
      return;
    }

    std::lock_guard<std::mutex> lock(mtx_);

    for (size_t i = 0; i < 6; ++i) {
      double v = static_cast<double>(msg->data[i]);
      if (qdes_in_degrees_) {
        v = v * kPi / 180.0;
      }
      latest_q_des_6_[i] = Clamp(v, q_min_6_[i], q_max_6_[i]);
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

      q_home_start_6_ = ExtractInput6FromMeasured17(current_jpos_meas_);
      q_target_safe_6_ = q_home_start_6_;
      track_start_6_ = q_home_start_6_;

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
  }

  std::vector<double> ExtractInput6FromMeasured17(
      const std::array<float, NUM_ARM_JOINTS> &q17_meas) const {
    std::vector<double> q6(6, 0.0);
    q6[0] = static_cast<double>(q17_meas[15]);
    q6[1] = static_cast<double>(q17_meas[16]);
    q6[2] = static_cast<double>(q17_meas[1]);
    q6[3] = static_cast<double>(q17_meas[3]);
    q6[4] = static_cast<double>(q17_meas[8]);
    q6[5] = static_cast<double>(q17_meas[10]);

    for (size_t i = 0; i < 6; ++i) {
      q6[i] = Clamp(q6[i], q_min_6_[i], q_max_6_[i]);
    }
    return q6;
  }

  void ApplyInput6ToDesired17(const std::vector<double> &q6,
                              const std::array<float, NUM_ARM_JOINTS> &base_q17,
                              std::array<float, NUM_ARM_JOINTS> &q17) {
    q17 = base_q17;

    q17[1] = static_cast<float>(q6[2]);
    q17[3] = static_cast<float>(q6[3]);
    q17[8] = static_cast<float>(q6[4]);
    q17[10] = static_cast<float>(q6[5]);
    q17[15] = static_cast<float>(q6[0]);
    q17[16] = static_cast<float>(q6[1]);
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
    if (!home_initialized_) {
      return BridgeMode::HOLD_CURRENT;
    }
    if (!home_reached_) {
      return BridgeMode::MOVE_TO_HOME;
    }
    if (!qdes_gate_open_) {
      qdes_gate_open_ = true;
      track_entry_start_time_ = this->now();
      track_start_6_ = q_home_6_;
      q_target_safe_6_ = q_home_6_;
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
      return q_home_6_;
    }

    const double elapsed = (now - home_transition_start_time_).seconds();
    const double move_duration = GetHomeMoveDuration();

    std::vector<double> q6(6, 0.0);
    for (size_t i = 0; i < 6; ++i) {
      const double dist = std::fabs(q_home_6_[i] - q_home_start_6_[i]);
      const double duration =
          (home_transition_velocity_ > 1e-6) ? dist / home_transition_velocity_ : 0.0;
      const double ti = (duration > 1e-6) ? Clamp(elapsed / duration, 0.0, 1.0) : 1.0;
      q6[i] = Lerp(q_home_start_6_[i], q_home_6_[i], ti);
    }

    if (elapsed >= move_duration) {
      if ((now - home_hold_start_time_).seconds() < control_dt_ * 1.5) {
        RCLCPP_INFO(this->get_logger(), "Reached q_home. Holding before enabling q_des tracking...");
      }
      if ((now - home_transition_start_time_).seconds() >= move_duration + home_hold_sec_) {
        home_reached_ = true;
        q_target_safe_6_ = q_home_6_;
        track_start_6_ = q_home_6_;
        track_entry_done_ = false;
        RCLCPP_INFO(this->get_logger(), "Home hold complete.");
      }
    } else {
      home_hold_start_time_ = now;
    }

    return q6;
  }

  std::vector<double> ComputeTrackEntryReference(const rclcpp::Time &now, bool qdes_fresh) {
    if (!qdes_fresh) {
      track_entry_done_ = false;
      return q_home_6_;
    }

    const double elapsed = (now - track_entry_start_time_).seconds();
    const double duration = std::max(control_dt_, track_entry_blend_sec_);
    const double t = Clamp(elapsed / duration, 0.0, 1.0);

    std::vector<double> target6 = latest_q_des_6_;
    for (size_t i = 0; i < 6; ++i) {
      target6[i] = Clamp(target6[i], q_min_6_[i], q_max_6_[i]);
      target6[i] = ema_alpha_ * target6[i] + (1.0 - ema_alpha_) * q_target_safe_6_[i];
    }

    std::vector<double> q6(6, 0.0);
    for (size_t i = 0; i < 6; ++i) {
      q6[i] = Lerp(track_start_6_[i], target6[i], t);
    }

    if (t >= 1.0) {
      track_entry_done_ = true;
    }

    return q6;
  }

  double GetHomeMoveDuration() const {
    double max_duration = 0.0;
    for (size_t i = 0; i < 6; ++i) {
      const double dist = std::fabs(q_home_6_[i] - q_home_start_6_[i]);
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
    std::vector<double> q_ref_6 = q_target_safe_6_;

    if (mode == BridgeMode::HOLD_CURRENT) {
      desired_17_ = current_jpos_meas_;
      weight_ = 0.0F;
    } else if (mode == BridgeMode::MOVE_TO_HOME) {
      q_ref_6 = ComputeHomeReference(now);
      q_target_safe_6_ = q_ref_6;
      used_max_delta = home_max_joint_delta_;
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput6ToDesired17(q_ref_6, base_q_17_, desired_17_);
    } else if (mode == BridgeMode::WAIT_FOR_QDES) {
      q_ref_6 = q_home_6_;
      q_target_safe_6_ = q_ref_6;
      track_start_6_ = q_home_6_;
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput6ToDesired17(q_ref_6, base_q_17_, desired_17_);
    } else if (mode == BridgeMode::TRACK_ENTRY) {
      used_max_delta = max_joint_delta_;
      q_ref_6 = ComputeTrackEntryReference(now, qdes_fresh);
      q_target_safe_6_ = q_ref_6;
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput6ToDesired17(q_target_safe_6_, base_q_17_, desired_17_);
    } else {
      q_ref_6 = latest_q_des_6_;
      for (size_t i = 0; i < 6; ++i) {
        q_ref_6[i] = Clamp(q_ref_6[i], q_min_6_[i], q_max_6_[i]);
        q_target_safe_6_[i] =
            ema_alpha_ * q_ref_6[i] + (1.0 - ema_alpha_) * q_target_safe_6_[i];
      }
      if (use_weight_ramp_) {
        weight_ = ClampF(weight_ + static_cast<float>(weight_acquire_rate_ * control_dt_),
                         0.0F, static_cast<float>(weight_active_));
      } else {
        weight_ = static_cast<float>(weight_active_);
      }
      ApplyInput6ToDesired17(q_target_safe_6_, base_q_17_, desired_17_);
    }

    StepTowardsDesired(used_max_delta);

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
      }
      RCLCPP_INFO(
          this->get_logger(),
          "[bridge] mode=%s gate=%s weight=%.2f fresh=%s | q6_deg: wr=%.1f wp=%.1f lsr=%.1f le=%.1f rsr=%.1f re=%.1f",
          mode_str,
          qdes_gate_open_ ? "open" : "closed",
          weight_,
          qdes_fresh ? "true" : "false",
          deg(q_target_safe_6_[0]),
          deg(q_target_safe_6_[1]),
          deg(q_target_safe_6_[2]),
          deg(q_target_safe_6_[3]),
          deg(q_target_safe_6_[4]),
          deg(q_target_safe_6_[5]));
      last_log_time_ = now;
    }
  }

 private:
  rclcpp::Publisher<LowCmd>::SharedPtr pub_arm_sdk_;
  rclcpp::Subscription<LowState>::SharedPtr sub_lowstate_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_qdes_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex mtx_;
  std::mutex shutdown_mtx_;

  std::string qdes_topic_;
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
  double shutdown_release_sec_;

  bool auto_move_to_home_;
  double home_hold_sec_;

  std::vector<double> q_home_6_;
  std::vector<double> q_min_6_;
  std::vector<double> q_max_6_;

  std::chrono::milliseconds sleep_time_{};

  LowState last_state_;
  std::array<G1Arm7JointIndex, NUM_ARM_JOINTS> arm_joints_;

  std::array<float, NUM_ARM_JOINTS> current_jpos_des_;
  std::array<float, NUM_ARM_JOINTS> current_jpos_meas_;
  std::array<float, NUM_ARM_JOINTS> desired_17_;
  std::array<float, NUM_ARM_JOINTS> base_q_17_;

  std::vector<double> latest_q_des_6_;
  std::vector<double> q_target_safe_6_;
  std::vector<double> q_home_start_6_;
  std::vector<double> track_start_6_;

  bool has_qdes_;
  bool has_lowstate_;
  bool shutdown_released_;
  bool home_initialized_;
  bool home_reached_;
  bool qdes_gate_open_;
  bool track_entry_done_{false};
  float weight_;

  rclcpp::Time last_qdes_time_;
  rclcpp::Time last_log_time_;
  rclcpp::Time home_transition_start_time_;
  rclcpp::Time home_hold_start_time_;
  rclcpp::Time track_entry_start_time_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<G1ArmSdkBridge>();

  rclcpp::on_shutdown([node]() {
    if (node) {
      node->ReleaseControlSafely();
    }
  });

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
