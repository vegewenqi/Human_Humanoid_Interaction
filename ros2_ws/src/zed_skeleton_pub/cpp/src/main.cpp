#include <sl/Camera.hpp>

#include "GLViewer.hpp"
#include "TrackingViewer.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters &param);
static inline bool valid2D(const sl::float2 &pt);
static inline bool valid3D(const sl::float3 &pt);
static inline float dist3D(const sl::float3 &a, const sl::float3 &b);
static inline double now_sec();

bool record_video = false;

int main(int argc, char **argv)
{
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::AUTO;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FORWARD;

    parseArgs(argc, argv, init_parameters);

    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS)
    {
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // --- ROS2 init ---
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("zed_skeleton_pub_node");
    auto pub_cloud = node->create_publisher<sensor_msgs::msg::PointCloud2>("/skeleton/points", 10);
    auto pub_conf  = node->create_publisher<std_msgs::msg::UInt8>("/skeleton/confidence", 10);
    auto pub_orient = node->create_publisher<std_msgs::msg::Float32MultiArray>("/skeleton/local_orientations", 10);
    RCLCPP_INFO(node->get_logger(), "Publishing /skeleton/points, /skeleton/confidence, /skeleton/local_orientations");

    PositionalTrackingParameters positional_tracking_parameters;
    positional_tracking_parameters.set_as_static = true;

    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS)
    {
        print("enable Positional Tracking", returned_state, "\nExit program.");
        zed.close();
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    BodyTrackingParameters body_tracker_params;
    body_tracker_params.enable_tracking = true;
    body_tracker_params.enable_body_fitting = true;
    body_tracker_params.body_format = sl::BODY_FORMAT::BODY_38;
    body_tracker_params.enable_segmentation = true;
    body_tracker_params.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_FAST;
    // body_tracker_params.allow_reduced_precision_inference = true;

    returned_state = zed.enableBodyTracking(body_tracker_params);
    if (returned_state != ERROR_CODE::SUCCESS)
    {
        print("enable Object Detection", returned_state, "\nExit program.");
        zed.close();
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;

    float image_aspect_ratio = camera_config.resolution.width / (1.f * camera_config.resolution.height);
    int requested_low_res_w = min(1280, (int)camera_config.resolution.width);
    sl::Resolution display_resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);

    cv::Mat image_left_ocv(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    Mat image_left(display_resolution, MAT_TYPE::U8_C4, image_left_ocv.data, image_left_ocv.step);
    sl::float2 img_scale(
        display_resolution.width / (float)camera_config.resolution.width,
        display_resolution.height / (float)camera_config.resolution.height);

    GLViewer viewer;
    viewer.init(argc, argv);

    Pose cam_pose;
    cam_pose.pose_data.setIdentity();

    BodyTrackingRuntimeParameters body_tracker_parameters_rt;
    body_tracker_parameters_rt.detection_confidence_threshold = 40;
    body_tracker_parameters_rt.skeleton_smoothing = 0.7;

    Bodies bodies;

    bool quit = false;
    string window_name = "ZEDposeDetect";
    int key_wait = 10;
    char key = ' ';

    using clock_t = std::chrono::high_resolution_clock;
    auto prev_time = clock_t::now();
    float fps = 0.f;

    std::vector<cv::Mat> frames_2D;
    std::vector<double> timestamps;

    while (!quit && rclcpp::ok())
    {
        auto err = zed.grab();
        if (err <= ERROR_CODE::SUCCESS)
        {
            zed.retrieveBodies(bodies, body_tracker_parameters_rt);

            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);

            viewer.updateData(bodies, cam_pose.pose_data);

            // Publish first tracked body (MVP): bodies.body_list[0]
            if (!bodies.body_list.empty()) {
                const auto &body = bodies.body_list[0];

                // PointCloud2 with 38 xyz points
                sensor_msgs::msg::PointCloud2 cloud;
                cloud.header.stamp = node->get_clock()->now();
                cloud.header.frame_id = "zed_world";
                cloud.height = 1;
                cloud.width = static_cast<uint32_t>(body.keypoint.size()); // should be 38 for BODY_38
                cloud.is_dense = false;

                sensor_msgs::PointCloud2Modifier modifier(cloud);
                modifier.setPointCloud2FieldsByString(1, "xyz");
                modifier.resize(cloud.width);

                sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
                sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
                sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

                for (size_t i = 0; i < body.keypoint.size(); ++i, ++iter_x, ++iter_y, ++iter_z) {
                    const sl::float3 &p = body.keypoint[i];
                    // Keep NaN/Inf as-is; subscriber can filter. Or set to NaN if invalid.
                    *iter_x = p.x;
                    *iter_y = p.y;
                    *iter_z = p.z;
                }

                pub_cloud->publish(cloud);

                // confidence (0-100)
                std_msgs::msg::UInt8 conf;
                // ZED SDK BodyData typically provides `confidence`
                conf.data = static_cast<uint8_t>(std::max(0, std::min(100, (int)body.confidence)));
                pub_conf->publish(conf);

                // local orientations: flatten as [x,y,z,w, x,y,z,w, ...] for BODY_38
                std_msgs::msg::Float32MultiArray orient_msg;
                orient_msg.data.reserve(body.local_orientation_per_joint.size() * 4);

                for (size_t i = 0; i < body.local_orientation_per_joint.size(); ++i) {
                    const auto &q = body.local_orientation_per_joint[i];
                    orient_msg.data.push_back(q.x);
                    orient_msg.data.push_back(q.y);
                    orient_msg.data.push_back(q.z);
                    orient_msg.data.push_back(q.w);
                }

                pub_orient->publish(orient_msg);
            }

            // Allow ROS2 to process callbacks
            rclcpp::spin_some(node);

            // printf("bodies is tracked %d \n", bodies.is_tracked);
            render_2D(image_left_ocv, img_scale, bodies.body_list, bodies.is_tracked);

            if (bodies.body_list.size() >= 2)
            {

                const auto &bodyA = bodies.body_list[0];
                const auto &bodyB = bodies.body_list[1];

                static const std::vector<int> joint_indices = {
                    10, 11, 12, 13, 14, 15, 16, 17};

                for (int idx : joint_indices)
                {

                    const sl::float2 &JA_2D = bodyA.keypoint_2d[idx];
                    const sl::float2 &JB_2D = bodyB.keypoint_2d[idx];

                    if (!valid2D(JA_2D) || !valid2D(JB_2D))
                    {
                        continue;
                    }

                    const sl::float3 &JA_3D = bodyA.keypoint[idx];
                    const sl::float3 &JB_3D = bodyB.keypoint[idx];

                    float dist = -1.f;
                    if (valid3D(JA_3D) && valid3D(JB_3D))
                    {
                        dist = dist3D(JA_3D, JB_3D);
                    }

                    cv::Point A((int)JA_2D.x, (int)JA_2D.y);
                    cv::Point B((int)JB_2D.x, (int)JB_2D.y);

                    cv::line(image_left_ocv, A, B, cv::Scalar(0, 255, 0), 2);
                    cv::circle(image_left_ocv, A, 4, cv::Scalar(0, 0, 255), -1);
                    cv::circle(image_left_ocv, B, 4, cv::Scalar(255, 0, 0), -1);
                }
            }

            auto now = clock_t::now();
            float dt = std::chrono::duration<float>(now - prev_time).count();
            prev_time = now;

            if (dt > 0.f)
            {
                fps = 1.f / dt;
            }

            cv::putText(
                image_left_ocv,
                std::to_string((int)fps) + " FPS",
                cv::Point(10, 70),
                cv::FONT_HERSHEY_COMPLEX,
                1.0,
                cv::Scalar(0, 0, 255),
                2);

            if (record_video == true)
            {
                cv::Mat frame_bgr;
                cv::cvtColor(image_left_ocv, frame_bgr, cv::COLOR_BGRA2BGR);
                frames_2D.push_back(frame_bgr.clone());
                timestamps.push_back(now_sec());

                std::vector<std::vector<sl::float3>> frame_points;
                for (const auto &body : bodies.body_list)
                {
                    std::vector<sl::float3> joints;
                    for (const auto &j : body.keypoint)
                        joints.push_back(j);
                    frame_points.push_back(joints);
                }
            }

            cv::imshow(window_name, image_left_ocv);

            key = cv::waitKey(key_wait);

            if (key == 'q')
            {
                quit = true;
            }
            if (key == 'p')
            {
                key_wait = (key_wait > 0) ? 0 : 10;
            }
            if (!viewer.isAvailable())
            {
                quit = true;
            }
        }
        else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            zed.setSVOPosition(0);
        }
        else
        {
            quit = true;
        }
    }

    double avg_fps = 0.0;
    if (record_video && timestamps.size() > 1)
    {
        std::vector<double> diffs;
        for (size_t i = 1; i < timestamps.size(); ++i)
            diffs.push_back(timestamps[i] - timestamps[i - 1]);

        double mean_dt = std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
        avg_fps = 1.0 / mean_dt;
    }
    if (record_video && !frames_2D.empty())
    {

        int w = frames_2D[0].cols;
        int h = frames_2D[0].rows;

        cv::VideoWriter video_out(
            "output.mp4",
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            avg_fps,
            cv::Size(w, h));

        for (const auto &f : frames_2D)
            video_out.write(f);

        video_out.release();
    }

    viewer.exit();
    image_left.free();
    bodies.body_list.clear();
    zed.disableBodyTracking();
    zed.disablePositionalTracking();
    zed.close();

    rclcpp::shutdown();

    return EXIT_SUCCESS;
}

static inline bool valid2D(const sl::float2 &pt)
{
    return pt.x >= 0.f && pt.y >= 0.f;
}

static inline bool valid3D(const sl::float3 &pt)
{
    return std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z);
}

static inline float dist3D(const sl::float3 &a, const sl::float3 &b)
{
    sl::float3 d = a - b;
    return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

static inline double now_sec()
{
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

void parseArgs(int argc, char **argv, InitParameters &param)
{
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos)
    {
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    }
    else if (argc > 1 && string(argv[1]).find(".svo") == string::npos)
    {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%u", &a, &b, &c, &d, &port) == 5)
        {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        }
        else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4)
        {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        }
        else if (arg.find("HD2K") != string::npos)
        {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        }
        else if (arg.find("HD1200") != string::npos)
        {
            param.camera_resolution = RESOLUTION::HD1200;
            cout << "[Sample] Using Camera in resolution HD1200" << endl;
        }
        else if (arg.find("HD1080") != string::npos)
        {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        }
        else if (arg.find("HD720") != string::npos)
        {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        }
        else if (arg.find("SVGA") != string::npos)
        {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "[Sample] Using Camera in resolution SVGA" << endl;
        }
        else if (arg.find("VGA") != string::npos)
        {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }

        if (arg.find("RECORD") != string::npos)
        {
            record_video = true;
            cout << "[Sample] Recording is enabled" << endl;
        }
    }
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix)
{
    cout << "[Sample]";
    if (err_code > ERROR_CODE::SUCCESS)
        cout << "[Error]";
    else if (err_code < ERROR_CODE::SUCCESS)
        cout << "[Warning]";
    cout << " " << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS)
    {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
