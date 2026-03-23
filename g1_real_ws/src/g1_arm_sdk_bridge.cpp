#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using namespace std::chrono_literals;

static const std::string kTopicArmSDK = "rt/arm_sdk";
static const std::string kTopicState  = "rt/lowstate";
constexpr double kPi = 3.14159265358979323846;

enum JointIndex {
    // Left leg
    kLeftHipPitch,
    kLeftHipRoll,
    kLeftHipYaw,
    kLeftKnee,
    kLeftAnkle,
    kLeftAnkleRoll,

    // Right leg
    kRightHipPitch,
    kRightHipRoll,
    kRightHipYaw,
    kRightKnee,
    kRightAnkle,
    kRightAnkleRoll,

    // Waist
    kWaistYaw,
    kWaistRoll,
    kWaistPitch,

    // Left arm
    kLeftShoulderPitch,
    kLeftShoulderRoll,
    kLeftShoulderYaw,
    kLeftElbow,
    kLeftWristRoll,
    kLeftWristPitch,
    kLeftWristYaw,

    // Right arm
    kRightShoulderPitch,
    kRightShoulderRoll,
    kRightShoulderYaw,
    kRightElbow,
    kRightWristRoll,
    kRightWristPitch,
    kRightWristYaw,

    kNotUsedJoint,
    kNotUsedJoint1,
    kNotUsedJoint2,
    kNotUsedJoint3,
    kNotUsedJoint4,
    kNotUsedJoint5
};

class G1ArmSdkBridge : public rclcpp::Node {
public:
    G1ArmSdkBridge() : Node("g1_arm_sdk_bridge") {
        // ---------------- parameters ----------------
        this->declare_parameter<std::string>("network_interface", "eth0");
        this->declare_parameter<std::string>("qdes_topic", "/g1_upperbody_q_des");
        this->declare_parameter<bool>("qdes_in_degrees", false);

        this->declare_parameter<double>("control_dt", 0.02);          // same as official SDK2 control loop period
        this->declare_parameter<double>("ema_alpha", 0.20);
        this->declare_parameter<double>("max_joint_velocity", 0.50);  // rad/s
        this->declare_parameter<double>("topic_timeout_sec", 0.30);

        this->declare_parameter<double>("kp_arm", 60.0);
        this->declare_parameter<double>("kd_arm", 1.5);
        this->declare_parameter<double>weight_("kp_waist", 40.0);
        this->declare_parameter<double>("kd_waist", 1.5);
        this->declare_parameter<double>("dq", 0.0);
        this->declare_parameter<double>("tau_ff", 0.0);

        this->declare_parameter<double>("weight_active", 1.0);
        this->declare_parameter<double>("weight_release_rate", 0.2);  // same as official example: decrease by 0.2 per second

        // 6 DOF home positions:
        // [waist_roll, waist_pitch, l_sh_roll, l_elbow, r_sh_roll, r_elbow]
        // home position: stand up straight with arms down
        // (-30-30, -30-30, -90-130, -60-120, -130,90, -60-120)
        // (-30-30, -30-30, 0-130, -60-120, -130,0, -60-120)
        this->declare_parameter<std::vector<double>>(
            "q_home_6", {0.0, 0.0, 0.0, 1.5708, 0.0, 1.5708});

        this->declare_parameter<std::vector<double>>(
            "q_min_6", {-0.52, -0.52, 0.0, -1.0472, -2.2515, -1.0472});

        this->declare_parameter<std::vector<double>>(
            "q_max_6", { 0.52, 0.52, 2.2515, 2.0944, 0.0, 2.0944});

        network_interface_ = this->get_parameter("network_interface").as_string();
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

        q_home_6_ = this->get_parameter("q_home_6").as_double_array();
        q_min_6_  = this->get_parameter("q_min_6").as_double_array();
        q_max_6_  = this->get_parameter("q_max_6").as_double_array();

        if (q_home_6_.size() != 6 || q_min_6_.size() != 6 || q_max_6_.size() != 6) {
            throw std::runtime_error("q_home_6 / q_min_6 / q_max_6 must all have length 6");
        }

        max_joint_delta_ = max_joint_velocity_ * control_dt_;

        arm_joints_ = {
            JointIndex::kLeftShoulderPitch, JointIndex::kLeftShoulderRoll,
            JointIndex::kLeftShoulderYaw,   JointIndex::kLeftElbow,
            JointIndex::kLeftWristRoll,     JointIndex::kLeftWristPitch,
            JointIndex::kLeftWristYaw,
            JointIndex::kRightShoulderPitch, JointIndex::kRightShoulderRoll,
            JointIndex::kRightShoulderYaw,   JointIndex::kRightElbow,
            JointIndex::kRightWristRoll,     JointIndex::kRightWristPitch,
            JointIndex::kRightWristYaw,
            JointIndex::kWaistYaw,
            JointIndex::kWaistRoll,
            JointIndex::kWaistPitch
        };

        // 17 DOF home positions: start with all zeros, only set the relevant joints to the original mapper's home positions
        target_pos_17_.fill(0.0f);
        current_jpos_des_.fill(0.0f);
        current_jpos_meas_.fill(0.0f);
        latest_q_des_6_.assign(6, 0.0);
        q_target_safe_6_ = q_home_6_;

        // mapping from 6 DOF input to 17 DOF target positions
        ApplyInput6ToTarget17(q_home_6_, target_pos_17_);

        // ---------------- ROS2 subscription ----------------
        sub_qdes_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            qdes_topic_, 10,
            std::bind(&G1ArmSdkBridge::OnQdes, this, std::placeholders::_1));

        // ---------------- SDK2 init ----------------
        unitree::robot::ChannelFactory::Instance()->Init(0, network_interface_);

        arm_sdk_publisher_.reset(
            new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(kTopicArmSDK));
        arm_sdk_publisher_->InitChannel();

        low_state_subscriber_.reset(
            new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(kTopicState));
        low_state_subscriber_->InitChannel(
            std::bind(&G1ArmSdkBridge::OnLowState, this, std::placeholders::_1), 1);

        last_qdes_time_ = this->now();
        last_log_time_ = this->now();

        weight_ = 0.0f;
        has_qdes_ = false;
        has_lowstate_ = false;
        started_control_ = false;

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(control_dt_),
            std::bind(&G1ArmSdkBridge::ControlLoop, this));

        RCLCPP_INFO(this->get_logger(), "G1ArmSdkBridge started.");
        RCLCPP_INFO(this->get_logger(), "network_interface = %s", network_interface_.c_str());
        RCLCPP_INFO(this->get_logger(), "qdes_topic        = %s", qdes_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "control_dt        = %.4f", control_dt_);
        RCLCPP_INFO(this->get_logger(), "max_joint_vel     = %.3f rad/s", max_joint_velocity_);
        RCLCPP_INFO(this->get_logger(), "waist is ENABLED in this version.");
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

    void OnLowState(const void *msg) {
        auto s = (const unitree_hg::msg::dds_::LowState_ *)msg;
        if (s == nullptr) return;

        std::lock_guard<std::mutex> lock(mtx_);
        std::memcpy(&state_msg_, s, sizeof(unitree_hg::msg::dds_::LowState_));

        for (size_t i = 0; i < arm_joints_.size(); ++i) {
            current_jpos_meas_.at(i) = state_msg_.motor_state().at(arm_joints_.at(i)).q();
        }

        if (!has_lowstate_) {
            // on the first received state, initialize the desired positions to the current measured positions to avoid jumps
            current_jpos_des_ = current_jpos_meas_;
            target_pos_17_ = current_jpos_meas_;
            has_lowstate_ = true;
        }
    }

    void ApplyInput6ToTarget17(const std::vector<double> &q6, std::array<float, 17> &q17) {
        // first, set all uncontrolled joints to a conservative home position
        q17.fill(0.0f);

        // 6 DOF input order:
        // [waist_roll, waist_pitch, l_sh_roll, l_elbow, r_sh_roll, r_elbow]

        // left arm
        q17.at(0)  = 0.0f;                 // left shoulder pitch
        q17.at(1)  = static_cast<float>(q6[2]); // left shoulder roll
        q17.at(2)  = 0.0f;                 // left shoulder yaw
        q17.at(3)  = static_cast<float>(q6[3]); // left elbow
        q17.at(4)  = 0.0f;                 // left wrist roll
        q17.at(5)  = 0.0f;                 // left wrist pitch
        q17.at(6)  = 0.0f;                 // left wrist yaw

        // right arm
        q17.at(7)  = 0.0f;                 // right shoulder pitch
        q17.at(8)  = static_cast<float>(q6[4]); // right shoulder roll
        q17.at(9)  = 0.0f;                 // right shoulder yaw
        q17.at(10) = static_cast<float>(q6[5]); // right elbow
        q17.at(11) = 0.0f;                 // right wrist roll
        q17.at(12) = 0.0f;                 // right wrist pitch
        q17.at(13) = 0.0f;                 // right wrist yaw

        // waist
        q17.at(14) = 0.0f;
        q17.at(15) = static_cast<float>(q6[0]); // waist roll
        q17.at(16) = static_cast<float>(q6[1]); // waist pitch
    }

    void FillLowCmdFromCurrentDes(unitree_hg::msg::dds_::LowCmd_ &msg) {
        for (size_t j = 0; j < arm_joints_.size(); ++j) {
            const int idx = arm_joints_.at(j);

            const bool is_waist = (idx == JointIndex::kWaistYaw ||
                                   idx == JointIndex::kWaistRoll ||
                                   idx == JointIndex::kWaistPitch);

            msg.motor_cmd().at(idx).q(current_jpos_des_.at(j));
            msg.motor_cmd().at(idx).dq(dq_);
            msg.motor_cmd().at(idx).kp(is_waist ? kp_waist_ : kp_arm_);
            msg.motor_cmd().at(idx).kd(is_waist ? kd_waist_ : kd_arm_);
            msg.motor_cmd().at(idx).tau(tau_ff_);
        }

        msg.motor_cmd().at(JointIndex::kNotUsedJoint).q(weight_);
    }

    void ControlLoop() {
        std::lock_guard<std::mutex> lock(mtx_);

        if (!has_lowstate_) {
            return;
        }

        const rclcpp::Time now = this->now();
        const bool qdes_fresh =
            has_qdes_ && ((now - last_qdes_time_).seconds() <= topic_timeout_sec_);

        std::vector<double> q_ref_6 = q_home_6_;
        if (qdes_fresh) {
            q_ref_6 = latest_q_des_6_;
        }

        // 6 DOF clip
        for (size_t i = 0; i < 6; ++i) {
            q_ref_6[i] = Clamp(q_ref_6[i], q_min_6_[i], q_max_6_[i]);
        }

        // 6 DOF EMA
        for (size_t i = 0; i < 6; ++i) {
            q_target_safe_6_[i] =
                ema_alpha_ * q_ref_6[i] + (1.0 - ema_alpha_) * q_target_safe_6_[i];
            q_target_safe_6_[i] = Clamp(q_target_safe_6_[i], q_min_6_[i], q_max_6_[i]);
        }

        // mapping to 17 DOF target positions
        std::array<float, 17> desired_17{};
        ApplyInput6ToTarget17(q_target_safe_6_, desired_17);

        // post control logic：new topic directly sets weight to active
        if (qdes_fresh) {
            weight_ = static_cast<float>(weight_active_);
            started_control_ = true;
        } else {
            // overtime, slowly release the weight according to the official example style
            const float delta_weight = static_cast<float>(weight_release_rate_ * control_dt_);
            weight_ -= delta_weight;
            weight_ = ClampF(weight_, 0.0f, static_cast<float>(weight_active_));
        }

        // Joint command limiting
        for (size_t j = 0; j < current_jpos_des_.size(); ++j) {
            float err = desired_17.at(j) - current_jpos_des_.at(j);
            err = ClampF(err,
                         -static_cast<float>(max_joint_delta_),
                         static_cast<float>(max_joint_delta_));
            current_jpos_des_.at(j) += err;
        }

        unitree_hg::msg::dds_::LowCmd_ msg;
        FillLowCmdFromCurrentDes(msg);
        arm_sdk_publisher_->Write(msg);

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
    // params
    std::string network_interface_;
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

    std::vector<double> q_home_6_;
    std::vector<double> q_min_6_;
    std::vector<double> q_max_6_;

    // ros
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_qdes_;
    rclcpp::TimerBase::SharedPtr timer_;

    // sdk2
    unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> arm_sdk_publisher_;
    unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> low_state_subscriber_;

    // state
    std::mutex mtx_;
    unitree_hg::msg::dds_::LowState_ state_msg_;

    std::array<JointIndex, 17> arm_joints_;
    std::array<float, 17> target_pos_17_;
    std::array<float, 17> current_jpos_des_;
    std::array<float, 17> current_jpos_meas_;

    std::vector<double> latest_q_des_6_;
    std::vector<double> q_target_safe_6_;

    bool has_qdes_;
    bool has_lowstate_;
    bool started_control_;
    float weight_;

    rclcpp::Time last_qdes_time_;
    rclcpp::Time last_log_time_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<G1ArmSdkBridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}