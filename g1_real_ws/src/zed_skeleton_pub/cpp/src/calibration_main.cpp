/*
这版做的事是：
从左图检测 AprilTag
只跟踪 tag36h11 家族里的 id=0
用四个角点 + OpenCV solvePnP 求 tag center 在相机系下的位置
用 ZED 当前相机 pose 转成 zed_world
发布 /tag_center_zed_world
 */
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

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

static cv::Point3f cameraToWorld(const sl::Pose& cam_pose, const cv::Point3f& p_cam)
{
    sl::Translation t = cam_pose.getTranslation();
    sl::Orientation q = cam_pose.getOrientation();

    float qx = q.x;
    float qy = q.y;
    float qz = q.z;
    float qw = q.w;

    cv::Matx33f R(
        1.0f - 2.0f * (qy * qy + qz * qz), 2.0f * (qx * qy - qz * qw),       2.0f * (qx * qz + qy * qw),
        2.0f * (qx * qy + qz * qw),       1.0f - 2.0f * (qx * qx + qz * qz), 2.0f * (qy * qz - qx * qw),
        2.0f * (qx * qz - qy * qw),       2.0f * (qy * qz + qx * qw),       1.0f - 2.0f * (qx * qx + qy * qy)
    );

    cv::Vec3f pc(p_cam.x, p_cam.y, p_cam.z);
    cv::Vec3f pw = R * pc + cv::Vec3f(t.x, t.y, t.z);

    return cv::Point3f(pw[0], pw[1], pw[2]);
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
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD1080;
    init_parameters.camera_fps = 60;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    init_parameters.coordinate_units = UNIT::METER;

    parseArgs(argc, argv, init_parameters);

    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS)
    {
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("zed_calibration_node");

    auto pub_tag_center = node->create_publisher<geometry_msgs::msg::PointStamped>(
        "/tag_center_zed_world", 10);

    RCLCPP_INFO(node->get_logger(), "Publishing /tag_center_zed_world (AprilTag mode)");

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

    Pose cam_pose;
    cam_pose.pose_data.setIdentity();

    bool quit = false;
    string window_name = "ZED Calibration AprilTag";

    while (!quit && rclcpp::ok())
    {
        auto err = zed.grab();
        if (err <= ERROR_CODE::SUCCESS)
        {
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);

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

                cv::Point3f p_world = cameraToWorld(cam_pose, p_cam);

                // debug print
                // RCLCPP_INFO_THROTTLE(
                //     node->get_logger(),
                //     *node->get_clock(),
                //     1000,
                //     "p_cam=[%.3f, %.3f, %.3f], cam_t=[%.3f, %.3f, %.3f], p_world=[%.3f, %.3f, %.3f]",
                //     p_cam.x, p_cam.y, p_cam.z,
                //     cam_pose.getTranslation().x,
                //     cam_pose.getTranslation().y,
                //     cam_pose.getTranslation().z,
                //     p_world.x, p_world.y, p_world.z
                // );

                geometry_msgs::msg::PointStamped msg;
                msg.header.stamp = node->get_clock()->now();
                msg.header.frame_id = "zed_world";
                msg.point.x = p_world.x;
                msg.point.y = p_world.y;
                msg.point.z = p_world.z;

                pub_tag_center->publish(msg);
                published = true;

                RCLCPP_INFO_THROTTLE(
                    node->get_logger(),
                    *node->get_clock(),
                    1000,
                    "Tag %d in zed_world: [%.3f, %.3f, %.3f]",
                    target_id, msg.point.x, msg.point.y, msg.point.z);

                cv::putText(
                    vis,
                    "world [" +
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