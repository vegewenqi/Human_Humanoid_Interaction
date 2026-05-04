/*
这版做的事是：
AprilTag 在某一个可见相机里检测
        ↓
solvePnP 得到 tag center in that camera frame
        ↓
用 ZED360 JSON 里的该相机 pose
        ↓
转成 fusion_world
        ↓
发布 /tag_center_fusion_world
 */
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <sl/Fusion.hpp>
#include "utils.hpp"

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/common/image_u8.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters &param);

static cv::Point3f transformPoint(const sl::Transform& T, const cv::Point3f& p)
{
    cv::Point3f out;
    out.x = T(0, 0) * p.x + T(0, 1) * p.y + T(0, 2) * p.z + T(0, 3);
    out.y = T(1, 0) * p.x + T(1, 1) * p.y + T(1, 2) * p.z + T(1, 3);
    out.z = T(2, 0) * p.x + T(2, 1) * p.y + T(2, 2) * p.z + T(2, 3);
    return out;
}

static std::vector<cv::Point3f> makeMarkerObjectPoints(float marker_size_m)
{
    const float s = marker_size_m * 0.5f;
    // Tag center at origin, z=0 plane
    return {
        cv::Point3f(-s, -s, 0.f),
        cv::Point3f( s, -s, 0.f),
        cv::Point3f( s,  s, 0.f),
        cv::Point3f(-s,  s, 0.f)
    };
}

int main(int argc, char **argv)
{
    std::string fusion_config_path = "/home/user/Documents/ZED/zed360_calibration.json";
    unsigned int tag_camera_serial = 41235597;

    if (argc >= 2) {
        fusion_config_path = argv[1];
    }

    if (argc >= 3) {
        tag_camera_serial = static_cast<unsigned int>(std::stoul(argv[2]));
    }

    std::cout << "Using fusion config: " << fusion_config_path << std::endl;
    std::cout << "Using tag camera serial: " << tag_camera_serial << std::endl;

    constexpr sl::COORDINATE_SYSTEM COORDINATE_SYSTEM_USED =
        sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    constexpr sl::UNIT UNIT_USED = sl::UNIT::METER;

    auto configurations = sl::readFusionConfigurationFile(
        fusion_config_path.c_str(),
        COORDINATE_SYSTEM_USED,
        UNIT_USED
    );

    if (configurations.empty()) {
        std::cerr << "Empty fusion config file." << std::endl;
        return EXIT_FAILURE;
    }

    bool found = false;
    sl::Transform T_cam_to_fusion;
    sl::InputType selected_input;

    for (auto& conf : configurations) {
        if (static_cast<unsigned int>(conf.serial_number) == tag_camera_serial) {
            found = true;
            T_cam_to_fusion = conf.pose;
            selected_input = conf.input_type;
            break;
        }
    }

    if (!found) {
        std::cerr << "Camera serial " << tag_camera_serial
                << " not found in fusion config." << std::endl;
        return EXIT_FAILURE;
    }

    Camera zed;
    InitParameters init_parameters;
    init_parameters.input = selected_input;
    init_parameters.camera_resolution = RESOLUTION::HD1080;
    init_parameters.camera_fps = 60;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = COORDINATE_SYSTEM_USED;
    init_parameters.coordinate_units = UNIT_USED;

    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("fusion_zed_calibration_node");

    auto pub_tag_center = node->create_publisher<geometry_msgs::msg::PointStamped>(
        "/tag_center_fusion_world", 10);

    RCLCPP_INFO(node->get_logger(), "Publishing /tag_center_fusion_world (AprilTag mode)");

    auto camera_config = zed.getCameraInformation().camera_configuration;
    auto calib = camera_config.calibration_parameters.left_cam;

    float image_aspect_ratio = camera_config.resolution.width / (1.f * camera_config.resolution.height);
    int requested_low_res_w = min(1280, (int)camera_config.resolution.width);
    sl::Resolution display_resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);

    cv::Mat image_left_ocv(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    Mat image_left(display_resolution, MAT_TYPE::U8_C4, image_left_ocv.data, image_left_ocv.step);

    float sx = display_resolution.width  / static_cast<float>(camera_config.resolution.width);
    float sy = display_resolution.height / static_cast<float>(camera_config.resolution.height);

    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
        calib.fx * sx, 0.0,         calib.cx * sx,
        0.0,         calib.fy * sy, calib.cy * sy,
        0.0,         0.0,           1.0);

    cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);

    const int target_id = 0;
    const float tag_size_m = 0.06f;

    apriltag_family_t* tf = tag36h11_create();
    apriltag_detector_t* td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = 2.0;
    td->quad_sigma = 0.0;
    td->nthreads = 2;
    td->debug = 0;
    td->refine_edges = 1;

    bool quit = false;
    string window_name = "ZED Calibration AprilTag";

    while (!quit && rclcpp::ok())
    {
        auto err = zed.grab();
        if (err <= ERROR_CODE::SUCCESS)
        {
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);

            cv::Mat bgr, gray, vis;
            cv::cvtColor(image_left_ocv, bgr, cv::COLOR_BGRA2BGR);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
            vis = bgr.clone();

            image_u8_t apriltag_img {
                .width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
            };

            zarray_t* detections = apriltag_detector_detect(td, &apriltag_img);

            bool published = false;

            for (int i = 0; i < zarray_size(detections); ++i)
            {
                apriltag_detection_t* det = nullptr;
                zarray_get(detections, i, &det);
                if (!det) continue;

                // Draw detection
                cv::line(vis,
                         cv::Point((int)det->p[0][0], (int)det->p[0][1]),
                         cv::Point((int)det->p[1][0], (int)det->p[1][1]),
                         cv::Scalar(0,255,0), 2);
                cv::line(vis,
                         cv::Point((int)det->p[1][0], (int)det->p[1][1]),
                         cv::Point((int)det->p[2][0], (int)det->p[2][1]),
                         cv::Scalar(0,255,0), 2);
                cv::line(vis,
                         cv::Point((int)det->p[2][0], (int)det->p[2][1]),
                         cv::Point((int)det->p[3][0], (int)det->p[3][1]),
                         cv::Scalar(0,255,0), 2);
                cv::line(vis,
                         cv::Point((int)det->p[3][0], (int)det->p[3][1]),
                         cv::Point((int)det->p[0][0], (int)det->p[0][1]),
                         cv::Scalar(0,255,0), 2);

                cv::putText(vis,
                            "id=" + std::to_string(det->id),
                            cv::Point((int)det->c[0], (int)det->c[1]),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.7,
                            cv::Scalar(0,255,0),
                            2);

                if (det->id != target_id)
                    continue;

                std::vector<cv::Point2f> image_points = {
                    cv::Point2f((float)det->p[0][0], (float)det->p[0][1]),
                    cv::Point2f((float)det->p[1][0], (float)det->p[1][1]),
                    cv::Point2f((float)det->p[2][0], (float)det->p[2][1]),
                    cv::Point2f((float)det->p[3][0], (float)det->p[3][1])
                };

                auto object_points = makeMarkerObjectPoints(tag_size_m);

                cv::Mat rvec, tvec;
                bool ok = cv::solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    false,
                    cv::SOLVEPNP_ITERATIVE);

                if (!ok)
                    continue;

                cv::drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, tag_size_m * 0.5);

                cv::Point3f p_cv(
                    static_cast<float>(tvec.at<double>(0,0)),
                    static_cast<float>(tvec.at<double>(1,0)),
                    static_cast<float>(tvec.at<double>(2,0))
                );
                // OpenCV camera frame: x right, y down, z forward
                // ZED RIGHT_HANDED_Z_UP_X_FWD: x forward, y left, z up
                cv::Point3f p_cam(
                    p_cv.z,
                    -p_cv.x,
                    -p_cv.y
                );

                cv::Point3f p_fusion = transformPoint(T_cam_to_fusion, p_cam);

                geometry_msgs::msg::PointStamped msg;
                msg.header.stamp = node->get_clock()->now();
                msg.header.frame_id = "fusion_world";
                msg.point.x = p_fusion.x;
                msg.point.y = p_fusion.y;
                msg.point.z = p_fusion.z;

                pub_tag_center->publish(msg);
                published = true;

                RCLCPP_INFO_THROTTLE(
                    node->get_logger(),
                    *node->get_clock(),
                    1000,
                    "Tag %d in fusion_world: [%.3f, %.3f, %.3f]",
                    target_id, msg.point.x, msg.point.y, msg.point.z);

                cv::putText(
                    vis,
                    "fusion [" +
                    std::to_string(msg.point.x).substr(0,5) + ", " +
                    std::to_string(msg.point.y).substr(0,5) + ", " +
                    std::to_string(msg.point.z).substr(0,5) + "]",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);
            }

            if (!published)
            {
                RCLCPP_INFO_THROTTLE(
                    node->get_logger(),
                    *node->get_clock(),
                    2000,
                    "Target AprilTag id=%d not detected", target_id);
            }

            apriltag_detections_destroy(detections);

            cv::imshow(window_name, vis);
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q')
            {
                quit = true;
            }

            rclcpp::spin_some(node);
        }
        else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            zed.setSVOPosition(0);
        }
        else
        {
            RCLCPP_ERROR(node->get_logger(), "ZED grab failed, exiting.");
            quit = true;
        }
    }

    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);

    image_left.free();
    zed.disablePositionalTracking();
    zed.close();

    rclcpp::shutdown();
    return EXIT_SUCCESS;
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
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        }
        else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4)
        {
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