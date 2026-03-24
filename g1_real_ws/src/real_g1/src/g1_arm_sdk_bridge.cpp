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
    this->declare_parameter<double>("ema_alpha", 0.20);
    this->declare_parameter<double>("max_joint_velocity", 0.50);
    this->declare_parameter<double>("topic_timeout_sec", 0.30);

    this->declare_parameter<double>("kp_arm", 60.0);
    this->declare_parameter<double>("kd_arm", 1.5);
    this->declare_parameter<double>("kp_waist", 40.0);
    this->declare_parameter<double>("kd_waist", 1.5);
    this->declare_parameter<double>("dq", 0.0);
    this->declare_parameter<double>("tau_ff", 0.0);

    this->declare_parameter<double>("weight_active", 1.0);
    this->declare_parameter<double>("weight_release_rate", 0.2);
    this->declare_parameter<double>("shutdown_release_sec", 2.0);

    // 输入顺序:
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
    weight_release_rate_ = this->get_parameter("weight_release_rate").as_double();
    shutdown_release_sec_ = this->get_parameter("shutdown_release_sec").as_double();

    q_home_6_ = this->get_parameter("q_home_6").as_double_array();
    q_min_6_ = this->get_parameter("q_min_6").as_double_array();
    q_max_6_ = this->get_parameter("q_max_6").as_double_array();

    if (q_home_6_.size() != 6 || q_min_6_.size() != 6 || q_max_6_.size() != 6) {
      throw std::runtime_error("q_home_6 / q_min_6 / q_max_6 must all have length 6");
    }

    max_joint_delta_ = max_joint_velocity_ * control_dt_;
    sleep_time_ =
        std::chrono::milliseconds(static_cast<int>(control_dt_ * 1000.0));

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

    latest_q_des_6_.assign(6, 0.0);
    q_target_safe_6_ = q_home_6_;

    current_jpos_des_.fill(0.0F);
    current_jpos_meas_.fill(0.0F);
    desired_17_.fill(0.0F);

    ApplyInput6ToDesired17(q_home_6_, desired_17_);

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

    has_qdes_ = false;
    has_lowstate_ = false;
    shutdown_released_ = false;
    weight_ = 0.0F;

    RCLCPP_INFO(this->get_logger(), "Pure ROS2 G1ArmSdkBridge started.");
    RCLCPP_INFO(this->get_logger(), "qdes_topic = %s", qdes_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "control_dt = %.4f", control_dt_);
    RCLCPP_INFO(this->get_logger(), "topic_timeout_sec = %.3f", topic_timeout_sec_);
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

    RCLCPP_WARN(this->get_logger(), "Shutdown detected. Releasing arm control safely...");

    if (timer_) {
      timer_->cancel();
    }

    const int steps =
        std::max(1, static_cast<int>(std::round(shutdown_release_sec_ / control_dt_)));

    for (int i = 0; i < steps; ++i) {
      {
        std::lock_guard<std::mutex> lock(mtx_);
        const float delta_w = static_cast<float>(weight_release_rate_ * control_dt_);
        weight_ -= delta_w;
        weight_ = ClampF(weight_, 0.0F, static_cast<float>(weight_active_));

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

    RCLCPP_WARN(this->get_logger(), "Arm control released. Safe shutdown complete.");
  }

 private:
  static double Clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(x, hi));
  }

  static float ClampF(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
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
      // 第一次收到状态，用当前实测值初始化命令，避免跳变
      current_jpos_des_ = current_jpos_meas_;
      desired_17_ = current_jpos_meas_;
      has_lowstate_ = true;
    }
  }

  void ApplyInput6ToDesired17(const std::vector<double> &q6,
                              std::array<float, NUM_ARM_JOINTS> &q17) {
    q17.fill(0.0F);

    // q6:
    // [waist_roll, waist_pitch, l_sh_roll, l_elbow, r_sh_roll, r_elbow]

    // left arm
    q17[0] = 0.0F;
    q17[1] = static_cast<float>(q6[2]);
    q17[2] = 0.0F;
    q17[3] = static_cast<float>(q6[3]);
    q17[4] = 0.0F;
    q17[5] = 0.0F;
    q17[6] = 0.0F;

    // right arm
    q17[7] = 0.0F;
    q17[8] = static_cast<float>(q6[4]);
    q17[9] = 0.0F;
    q17[10] = static_cast<float>(q6[5]);
    q17[11] = 0.0F;
    q17[12] = 0.0F;
    q17[13] = 0.0F;

    // waist
    q17[14] = 0.0F;
    q17[15] = static_cast<float>(q6[0]);
    q17[16] = static_cast<float>(q6[1]);
  }

  void FillLowCmdFromCurrentDes(LowCmd &cmd) {
    for (size_t j = 0; j < arm_joints_.size(); ++j) {
      int idx = static_cast<int>(arm_joints_[j]);
      bool is_waist =
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

    std::vector<double> q_ref_6 = q_home_6_;
    if (qdes_fresh) {
      q_ref_6 = latest_q_des_6_;
    }

    for (size_t i = 0; i < 6; ++i) {
      q_ref_6[i] = Clamp(q_ref_6[i], q_min_6_[i], q_max_6_[i]);
    }

    for (size_t i = 0; i < 6; ++i) {
      q_target_safe_6_[i] =
          ema_alpha_ * q_ref_6[i] + (1.0 - ema_alpha_) * q_target_safe_6_[i];
      q_target_safe_6_[i] = Clamp(q_target_safe_6_[i], q_min_6_[i], q_max_6_[i]);
    }

    ApplyInput6ToDesired17(q_target_safe_6_, desired_17_);

    if (qdes_fresh) {
      weight_ = static_cast<float>(weight_active_);
    } else {
      const float delta_w = static_cast<float>(weight_release_rate_ * control_dt_);
      weight_ -= delta_w;
      weight_ = ClampF(weight_, 0.0F, static_cast<float>(weight_active_));
    }

    for (size_t j = 0; j < current_jpos_des_.size(); ++j) {
      float err = desired_17_[j] - current_jpos_des_[j];
      err = ClampF(err,
                   -static_cast<float>(max_joint_delta_),
                   static_cast<float>(max_joint_delta_));
      current_jpos_des_[j] += err;
    }

    LowCmd cmd;
    FillLowCmdFromCurrentDes(cmd);
    pub_arm_sdk_->publish(cmd);

    if ((now - last_log_time_).seconds() > 1.0) {
      auto deg = [](double rad) { return rad * 180.0 / kPi; };
      RCLCPP_INFO(
          this->get_logger(),
          "[bridge] weight=%.2f fresh=%s | q6_deg: wr=%.1f wp=%.1f lsr=%.1f le=%.1f rsr=%.1f re=%.1f",
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

  double control_dt_;
  double ema_alpha_;
  double max_joint_velocity_;
  double max_joint_delta_;
  double topic_timeout_sec_;

  double kp_arm_;
  double kd_arm_;
  double kp_waist_;
  double kd_waist_;
  double dq_;
  double tau_ff_;

  double weight_active_;
  double weight_release_rate_;
  double shutdown_release_sec_;

  std::vector<double> q_home_6_;
  std::vector<double> q_min_6_;
  std::vector<double> q_max_6_;

  std::chrono::milliseconds sleep_time_{};

  LowState last_state_;
  std::array<G1Arm7JointIndex, NUM_ARM_JOINTS> arm_joints_;
  std::array<float, NUM_ARM_JOINTS> current_jpos_des_;
  std::array<float, NUM_ARM_JOINTS> current_jpos_meas_;
  std::array<float, NUM_ARM_JOINTS> desired_17_;

  std::vector<double> latest_q_des_6_;
  std::vector<double> q_target_safe_6_;

  bool has_qdes_;
  bool has_lowstate_;
  bool shutdown_released_;
  float weight_;

  rclcpp::Time last_qdes_time_;
  rclcpp::Time last_log_time_;
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