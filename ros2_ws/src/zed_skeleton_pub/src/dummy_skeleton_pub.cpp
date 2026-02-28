#include <chrono>
#include <vector>
#include <cstring>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/u_int8.hpp"

using namespace std::chrono_literals;

class DummySkeletonPub : public rclcpp::Node {
public:
  DummySkeletonPub() : Node("dummy_skeleton_pub") {
    pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/skeleton/points", 10);
    pub_conf_  = this->create_publisher<std_msgs::msg::UInt8>("/skeleton/confidence", 10);
    timer_ = this->create_wall_timer(33ms, std::bind(&DummySkeletonPub::tick, this));
    phase_ = 0.0f;
    RCLCPP_INFO(get_logger(), "Dummy publisher started: /skeleton/points and /skeleton/confidence");
  }

private:
  void tick() {
    // Create a PointCloud2 with 38 XYZ points.
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = this->get_clock()->now();
    cloud.header.frame_id = "zed_world";
    cloud.height = 1;
    cloud.width = 38;
    cloud.is_dense = false;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(cloud.width);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

    // Simple moving pattern: all joints on a line with small sinusoidal z.
    for (size_t i = 0; i < 38; ++i, ++iter_x, ++iter_y, ++iter_z) {
      *iter_x = static_cast<float>(i) * 0.01f;
      *iter_y = 0.0f;
      *iter_z = 0.2f + 0.05f * std::sin(phase_ + static_cast<float>(i) * 0.1f);
    }
    phase_ += 0.1f;

    std_msgs::msg::UInt8 conf;
    conf.data = 80;

    pub_cloud_->publish(cloud);
    pub_conf_->publish(conf);
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr pub_conf_;
  rclcpp::TimerBase::SharedPtr timer_;
  float phase_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DummySkeletonPub>());
  rclcpp::shutdown();
  return 0;
}