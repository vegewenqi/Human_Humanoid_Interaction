#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include "FusionGLViewer.hpp"
#include "TrackingViewer.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <numeric>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
static inline bool valid2D(const sl::float2 &pt);
static inline bool valid3D(const sl::float3 &pt);
static inline float dist3D(const sl::float3 &a, const sl::float3 &b);
static inline int selectBestBodyIndex(const sl::Bodies &bodies);
static bool hasArg(int argc, char **argv, const std::string &flag);
static std::string getFirstNonFlagArg(int argc, char **argv);
static inline sl::float3 transformPoint(const sl::Transform &T, const sl::float3 &p);
static inline sl::Transform inverseRigidTransform(const sl::Transform &T);

bool record_video = false;

struct Trigger {
    std::mutex mtx;
    std::condition_variable cv;
    bool running = true;
    std::map<unsigned int, bool> states;

    void notifyZED() {
        std::unique_lock<std::mutex> lock(mtx);
        for (auto &s : states) {
            s.second = false;
        }
        cv.notify_all();
    }
};

class LocalZedCamera {
public:
    LocalZedCamera() = default;

    ~LocalZedCamera() {
        stop();
        if (zed.isOpened()) {
            zed.close();
        }
    }

    bool open(
        const sl::InputType &input,
        Trigger *trigger_ref,
        int sdk_gpu_id,
        const sl::InitParameters &base_init_params
    ) {
        trigger = trigger_ref;

        sl::InitParameters init_parameters = base_init_params;
        init_parameters.input = input;
        init_parameters.sdk_gpu_id = sdk_gpu_id;

        auto returned_state = zed.open(init_parameters);
        if (returned_state > ERROR_CODE::SUCCESS) {
            print("Open Camera", returned_state, "\nSkip this camera.");
            return false;
        }

        serial = zed.getCameraInformation().serial_number;
        trigger->states[serial] = false;

        PositionalTrackingParameters positional_tracking_parameters;
        positional_tracking_parameters.set_as_static = true;

        returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
        if (returned_state != ERROR_CODE::SUCCESS) {
            print("enable Positional Tracking", returned_state, "\nSkip this camera.");
            zed.close();
            return false;
        }

        BodyTrackingParameters body_tracker_params;
        body_tracker_params.enable_tracking = true;
        body_tracker_params.enable_body_fitting = false;
        body_tracker_params.body_format = sl::BODY_FORMAT::BODY_38;
        body_tracker_params.enable_segmentation = false;
        body_tracker_params.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
        body_tracker_params.allow_reduced_precision_inference = true;

        returned_state = zed.enableBodyTracking(body_tracker_params);
        if (returned_state != ERROR_CODE::SUCCESS) {
            print("enable Body Tracking", returned_state, "\nSkip this camera.");
            zed.close();
            return false;
        }

        // This is SDK internal publishing for Fusion, not ROS publishing.
        returned_state = zed.startPublishing();
        if (returned_state != ERROR_CODE::SUCCESS) {
            print("startPublishing", returned_state, "\nSkip this camera.");
            zed.close();
            return false;
        }

        opened = true;
        return true;
    }

    void start() {
        if (opened && zed.isOpened()) {
            runner = std::thread(&LocalZedCamera::work, this);
        }
    }

    void stop() {
        if (runner.joinable()) {
            runner.join();
        }
    }

    unsigned int getSerial() const {
        return serial;
    }

    bool isOpened() const {
        return opened;
    }

    sl::CameraInformation getCameraInformation() {
        return zed.getCameraInformation();
    }

private:
    void work() {
        sl::RuntimeParameters rt;

        while (trigger && trigger->running) {
            std::unique_lock<std::mutex> lk(trigger->mtx);
            trigger->cv.wait(lk);

            if (!trigger->running) {
                break;
            }

            if (zed.grab(rt) <= ERROR_CODE::SUCCESS) {
                // Fusion retrieves the published body data internally.
            }

            trigger->states[serial] = true;
        }
    }

private:
    sl::Camera zed;
    Trigger *trigger = nullptr;
    std::thread runner;
    unsigned int serial = 0;
    bool opened = false;
};

int main(int argc, char **argv)
{
    std::string fusion_config_path = getFirstNonFlagArg(argc, argv);

    if (fusion_config_path.empty()) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <zed360_calibration.json> [--show-3d] [--hide-2d] [--hide-all-viewer] [--publish-ref-zed-world]\n";
        return EXIT_FAILURE;
    }

    bool show_2d_viewer = true;
    bool show_3d_viewer = false;

    // Default: publish fused skeleton in fusion_world.
    // Add --publish-ref-zed-world to transform fused skeleton back to the
    // reference ZED frame for compatibility with the old sim transform.
    bool publish_ref_zed_world = hasArg(argc, argv, "--publish-ref-zed-world");

    // Reference camera serial used as the old zed_world-compatible frame.
    // Keep this fixed unless intentionally change the reference camera.
    unsigned int reference_zed_serial = 41235597;

    if (hasArg(argc, argv, "--show-3d")) {
        show_3d_viewer = true;
    }

    if (hasArg(argc, argv, "--hide-2d")) {
        show_2d_viewer = false;
    }

    if (hasArg(argc, argv, "--hide-all-viewer")) {
        show_2d_viewer = false;
        show_3d_viewer = false;
    }

    // ---------------------------------------------------------------------
    // Camera parameters
    // ---------------------------------------------------------------------
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::SVGA;
    init_parameters.camera_fps = 30;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL_LIGHT;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;

    // ---------------------------------------------------------------------
    // ROS2 init
    // ---------------------------------------------------------------------
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("zed_fusion_skeleton_pub_node");

    auto pub_cloud = node->create_publisher<sensor_msgs::msg::PointCloud2>("/skeleton/points", 10);
    auto pub_conf  = node->create_publisher<std_msgs::msg::UInt8>("/skeleton/confidence", 10);
    auto pub_orient = node->create_publisher<std_msgs::msg::Float32MultiArray>("/skeleton/local_orientations", 10);

    RCLCPP_INFO(node->get_logger(), "Publishing /skeleton/points, /skeleton/confidence");
    RCLCPP_INFO(node->get_logger(), "Fusion config: %s", fusion_config_path.c_str());
    RCLCPP_INFO(node->get_logger(), "2D viewer: %s", show_2d_viewer ? "ON" : "OFF");
    RCLCPP_INFO(node->get_logger(), "3D viewer: %s", show_3d_viewer ? "ON" : "OFF");
    RCLCPP_INFO(node->get_logger(), "Publish frame mode: %s",
                publish_ref_zed_world ? "fusion_in_zed_world" : "fusion_world");
    RCLCPP_INFO(node->get_logger(), "Reference ZED serial: %u", reference_zed_serial);

    // ---------------------------------------------------------------------
    // Read ZED360 calibration / fusion configuration
    // ---------------------------------------------------------------------
    constexpr sl::COORDINATE_SYSTEM COORDINATE_SYSTEM_USED =
        sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
    constexpr sl::UNIT UNIT_USED = sl::UNIT::METER;

    auto configurations = sl::readFusionConfigurationFile(
        fusion_config_path.c_str(),
        COORDINATE_SYSTEM_USED,
        UNIT_USED
    );

    if (configurations.empty()) {
        RCLCPP_ERROR(node->get_logger(), "Empty or invalid fusion configuration file.");
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    RCLCPP_INFO(node->get_logger(), "Loaded %zu camera configuration(s).", configurations.size());

    // ---------------------------------------------------------------------
    // Optional transform: fusion_world -> reference ZED frame
    // ---------------------------------------------------------------------
    sl::Transform T_ref_zed_to_fusion;
    sl::Transform T_fusion_to_ref_zed;
    bool found_reference_zed = false;

    for (auto &conf : configurations) {
        if (static_cast<unsigned int>(conf.serial_number) == reference_zed_serial) {
            T_ref_zed_to_fusion = conf.pose;
            T_fusion_to_ref_zed = inverseRigidTransform(T_ref_zed_to_fusion);
            found_reference_zed = true;
            break;
        }
    }

    if (publish_ref_zed_world && !found_reference_zed) {
        RCLCPP_ERROR(
            node->get_logger(),
            "Reference ZED serial %u not found in ZED360 config.",
            reference_zed_serial
        );
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    // ---------------------------------------------------------------------
    // Open all local cameras directly in this main.cpp
    // ---------------------------------------------------------------------
    Trigger trigger;

    std::vector<std::unique_ptr<LocalZedCamera>> cameras_local;
    std::vector<int> opened_serials;

    int cam_idx = 0;
    int gpu_id = 0;

    for (auto &conf : configurations) {
        if (conf.communication_parameters.getType()
            != sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS) {
            continue;
        }

        std::cout << "Try to open ZED " << conf.serial_number << ".." << std::flush;

        auto cam = std::make_unique<LocalZedCamera>();
        bool ok = cam->open(conf.input_type, &trigger, gpu_id, init_parameters);

        if (!ok) {
            std::cerr << " failed. Skip." << std::endl;
            cam_idx++;
            continue;
        }

        opened_serials.push_back(conf.serial_number);
        cameras_local.push_back(std::move(cam));

        std::cout << " ready!" << std::endl;

        cam_idx++;
    }

    if (cameras_local.empty()) {
        RCLCPP_ERROR(node->get_logger(), "No ZED camera could be opened.");
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    for (auto &cam : cameras_local) {
        cam->start();
    }

    // ---------------------------------------------------------------------
    // Prepare 2D viewer from the first camera configuration
    // ---------------------------------------------------------------------
    sl::Resolution display_resolution;
    sl::float2 img_scale(1.f, 1.f);
    cv::Mat image_left_ocv;
    sl::Mat image_left;

    if (show_2d_viewer) {
        auto camera_config = cameras_local[0]->getCameraInformation().camera_configuration;

        float image_aspect_ratio =
            camera_config.resolution.width / (1.f * camera_config.resolution.height);

        int requested_low_res_w = std::min(1280, (int)camera_config.resolution.width);

        display_resolution = sl::Resolution(
            requested_low_res_w,
            requested_low_res_w / image_aspect_ratio
        );

        image_left_ocv = cv::Mat(
            display_resolution.height,
            display_resolution.width,
            CV_8UC4,
            1
        );

        image_left = sl::Mat(
            display_resolution,
            MAT_TYPE::U8_C4,
            image_left_ocv.data,
            image_left_ocv.step
        );

        img_scale = sl::float2(
            display_resolution.width / (float)camera_config.resolution.width,
            display_resolution.height / (float)camera_config.resolution.height
        );
    }

    // ---------------------------------------------------------------------
    // Init Fusion
    // ---------------------------------------------------------------------
    sl::InitFusionParameters fusion_init_parameters;
    fusion_init_parameters.coordinate_units = UNIT_USED;
    fusion_init_parameters.coordinate_system = COORDINATE_SYSTEM_USED;

    // Keep this from the official sample. It only limits Fusion internal work.
    fusion_init_parameters.maximum_working_resolution = sl::Resolution(512, 360);

    sl::Fusion fusion;
    auto fusion_state = fusion.init(fusion_init_parameters);

    if (fusion_state != sl::FUSION_ERROR_CODE::SUCCESS) {
        RCLCPP_ERROR(node->get_logger(), "Fusion init failed.");

        trigger.running = false;
        trigger.notifyZED();

        for (auto &cam : cameras_local) {
            cam->stop();
        }

        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    std::vector<sl::CameraIdentifier> fusion_camera_ids;

    for (auto &conf : configurations) {
        bool opened = false;

        for (auto sn : opened_serials) {
            if (sn == conf.serial_number) {
                opened = true;
                break;
            }
        }

        if (!opened) {
            continue;
        }

        sl::CameraIdentifier uuid(conf.serial_number);

        auto sub_state = fusion.subscribe(
            uuid,
            conf.communication_parameters,
            conf.pose,
            conf.override_gravity
        );

        if (sub_state != sl::FUSION_ERROR_CODE::SUCCESS) {
            std::cout << "Unable to subscribe to "
                      << std::to_string(uuid.sn)
                      << " . " << sub_state << std::endl;
        } else {
            std::cout << "Fusion subscribed to camera "
                      << std::to_string(uuid.sn) << std::endl;
            fusion_camera_ids.push_back(uuid);
        }
    }

    if (fusion_camera_ids.empty()) {
        RCLCPP_ERROR(node->get_logger(), "No cameras subscribed to Fusion.");

        trigger.running = false;
        trigger.notifyZED();

        for (auto &cam : cameras_local) {
            cam->stop();
        }

        fusion.close();
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    // ---------------------------------------------------------------------
    // Fusion body tracking parameters
    // ---------------------------------------------------------------------
    sl::BodyTrackingFusionParameters body_fusion_params;
    body_fusion_params.enable_tracking = true;
    body_fusion_params.enable_body_fitting = false;

    fusion_state = fusion.enableBodyTracking(body_fusion_params);

    if (fusion_state != sl::FUSION_ERROR_CODE::SUCCESS) {
        RCLCPP_ERROR(node->get_logger(), "Fusion enableBodyTracking failed.");

        trigger.running = false;
        trigger.notifyZED();

        for (auto &cam : cameras_local) {
            cam->stop();
        }

        fusion.close();
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    BodyTrackingFusionRuntimeParameters body_tracker_parameters_rt;
    body_tracker_parameters_rt.skeleton_minimum_allowed_keypoints = 7;
    body_tracker_parameters_rt.skeleton_minimum_allowed_camera = 1;
    body_tracker_parameters_rt.skeleton_smoothing = 0.7;

    // ---------------------------------------------------------------------
    // Optional 3D viewer
    // ---------------------------------------------------------------------
    std::unique_ptr<FusionGLViewer> viewer_3d;
    Pose cam_pose;
    cam_pose.pose_data.setIdentity();

    if (show_3d_viewer) {
        viewer_3d = std::make_unique<FusionGLViewer>();
        viewer_3d->init(argc, argv);
    }

    // ---------------------------------------------------------------------
    // Runtime variables
    // ---------------------------------------------------------------------
    Bodies fused_bodies;
    Bodies raw_bodies_for_2d;
    Bodies single_body;

    // Used only by the official Fusion 3D viewer.
    std::map<sl::CameraIdentifier, sl::Bodies> camera_raw_data;
    sl::FusionMetrics metrics;

    std::string window_name = "ZEDposeDetect";
    int key_wait = 10;
    char key = ' ';
    int frame_id = 0;

    using clock_t = std::chrono::high_resolution_clock;
    auto prev_time = clock_t::now();
    float fps = 0.f;

    std::vector<cv::Mat> frames_2D;
    std::vector<double> timestamps;

    bool quit = false;

    while (!quit && rclcpp::ok())
    {
        trigger.notifyZED();

        if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS)
        {
            frame_id++;

            fusion.retrieveBodies(fused_bodies, body_tracker_parameters_rt);

            // If 3D Fusion viewer is enabled, also retrieve each camera's raw body data,
            // camera pose, and fusion metrics for the official Fusion viewer.
            if (show_3d_viewer && viewer_3d) {
                for (auto &id : fusion_camera_ids) {
                    fusion.retrieveBodies(
                        camera_raw_data[id],
                        body_tracker_parameters_rt,
                        id
                    );

                    sl::Pose pose;
                    if (fusion.getPosition(
                            pose,
                            sl::REFERENCE_FRAME::WORLD,
                            id,
                            sl::POSITION_TYPE::RAW
                        ) == sl::POSITIONAL_TRACKING_STATE::OK) {
                        viewer_3d->setCameraPose(id.sn, pose.pose_data);
                    }
                }

                fusion.getProcessMetrics(metrics);
            }

            int best_idx = selectBestBodyIndex(fused_bodies);

            single_body.body_list.clear();
            single_body.is_new = fused_bodies.is_new;
            single_body.is_tracked = fused_bodies.is_tracked;

            if (best_idx >= 0)
            {
                single_body.body_list.push_back(fused_bodies.body_list[best_idx]);
            }

            // -----------------------------------------------------------------
            // Publish points / confidence every frame
            // -----------------------------------------------------------------
            if (!single_body.body_list.empty()) {
                const auto &body = single_body.body_list[0];

                sensor_msgs::msg::PointCloud2 cloud;
                cloud.header.stamp = node->get_clock()->now();

                // Default: fusion_world. Optional compatibility mode: reference ZED frame.
                cloud.header.frame_id = publish_ref_zed_world ? "fusion_in_zed_world" : "fusion_world";

                cloud.height = 1;
                cloud.width = static_cast<uint32_t>(body.keypoint.size());
                cloud.is_dense = false;

                sensor_msgs::PointCloud2Modifier modifier(cloud);
                modifier.setPointCloud2FieldsByString(1, "xyz");
                modifier.resize(cloud.width);

                sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
                sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
                sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

                for (size_t i = 0; i < body.keypoint.size(); ++i, ++iter_x, ++iter_y, ++iter_z) {
                    const sl::float3 &p_fusion = body.keypoint[i];

                    sl::float3 p_pub = p_fusion;
                    if (publish_ref_zed_world) {
                        p_pub = transformPoint(T_fusion_to_ref_zed, p_fusion);
                    }

                    *iter_x = p_pub.x;
                    *iter_y = p_pub.y;
                    *iter_z = p_pub.z;
                }

                pub_cloud->publish(cloud);

                std_msgs::msg::UInt8 conf;
                conf.data = static_cast<uint8_t>(
                    std::max(0, std::min(100, (int)body.confidence))
                );
                pub_conf->publish(conf);

                // Keep the old orientation publisher disabled.
                // Since enable_body_fitting=false, local orientations may be empty.
            }

            rclcpp::spin_some(node);

            // -----------------------------------------------------------------
            // Render display at lower frame rate
            // -----------------------------------------------------------------
            if (frame_id % 5 == 0)
            {
                if (show_2d_viewer && !fusion_camera_ids.empty()) {
                    auto main_cam_id = fusion_camera_ids[0];  // Just render the first camera's view for the 2D viewer.

                    auto state_view = fusion.retrieveImage(
                        image_left,
                        main_cam_id,
                        display_resolution
                    );

                    fusion.retrieveBodies(
                        raw_bodies_for_2d,
                        body_tracker_parameters_rt,
                        main_cam_id
                    );

                    Bodies raw_single_body_2d;
                    raw_single_body_2d.is_new = raw_bodies_for_2d.is_new;
                    raw_single_body_2d.is_tracked = raw_bodies_for_2d.is_tracked;

                    int raw_best_idx = selectBestBodyIndex(raw_bodies_for_2d);
                    if (raw_best_idx >= 0) {
                        raw_single_body_2d.body_list.push_back(
                            raw_bodies_for_2d.body_list[raw_best_idx]
                        );
                    }

                    if (state_view == sl::FUSION_ERROR_CODE::SUCCESS) {
                        // cam0 image + cam0 raw 2D skeleton overlay
                        render_2D(
                            image_left_ocv,
                            img_scale,
                            raw_single_body_2d.body_list,
                            raw_single_body_2d.is_tracked
                        );

                        auto now = clock_t::now();
                        float dt = std::chrono::duration<float>(now - prev_time).count();
                        prev_time = now;

                        if (dt > 0.f) {
                            fps = 1.f / dt;
                        }

                        cv::putText(
                            image_left_ocv,
                            std::to_string((int)fps) + " FPS",
                            cv::Point(10, 70),
                            cv::FONT_HERSHEY_COMPLEX,
                            1.0,
                            cv::Scalar(0, 0, 255),
                            2
                        );

                        if (record_video == true) {
                            cv::Mat frame_bgr;
                            cv::cvtColor(image_left_ocv, frame_bgr, cv::COLOR_BGRA2BGR);
                            frames_2D.push_back(frame_bgr.clone());
                            timestamps.push_back(
                                std::chrono::duration<double>(
                                    std::chrono::high_resolution_clock::now().time_since_epoch()
                                ).count()
                            );
                        }

                        cv::imshow(window_name, image_left_ocv);
                    }
                }

                if (show_3d_viewer && viewer_3d) {
                    viewer_3d->updateBodies(
                        fused_bodies,
                        camera_raw_data,
                        metrics
                    );

                    if (!viewer_3d->isAvailable()) {
                        quit = true;
                    }
                }

                if (show_2d_viewer) {
                    key = cv::waitKey(key_wait);

                    if (key == 'q' || key == 'Q') {
                        quit = true;
                    }
                }
            }
        }
    }

    if (show_3d_viewer && viewer_3d) {
        viewer_3d->exit();
    }

    if (show_2d_viewer) {
        cv::destroyAllWindows();
    }

    trigger.running = false;
    trigger.notifyZED();

    for (auto &cam : cameras_local) {
        cam->stop();
    }

    fusion.disableBodyTracking();
    fusion.close();

    rclcpp::shutdown();

    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix)
{
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout << " ";

    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << " | " << toString(err_code) << " : ";

    cout << msg_suffix << endl;
}

static inline sl::float3 transformPoint(const sl::Transform &T, const sl::float3 &p)
{
    sl::float3 out;
    out.x = T(0, 0) * p.x + T(0, 1) * p.y + T(0, 2) * p.z + T(0, 3);
    out.y = T(1, 0) * p.x + T(1, 1) * p.y + T(1, 2) * p.z + T(1, 3);
    out.z = T(2, 0) * p.x + T(2, 1) * p.y + T(2, 2) * p.z + T(2, 3);
    return out;
}

static inline sl::Transform inverseRigidTransform(const sl::Transform &T)
{
    sl::Transform inv;
    inv.setIdentity();

    // R_inv = R^T
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            inv(r, c) = T(c, r);
        }
    }

    // t_inv = -R^T * t
    inv(0, 3) = -(inv(0, 0) * T(0, 3) + inv(0, 1) * T(1, 3) + inv(0, 2) * T(2, 3));
    inv(1, 3) = -(inv(1, 0) * T(0, 3) + inv(1, 1) * T(1, 3) + inv(1, 2) * T(2, 3));
    inv(2, 3) = -(inv(2, 0) * T(0, 3) + inv(2, 1) * T(1, 3) + inv(2, 2) * T(2, 3));

    return inv;
}

static inline bool valid2D(const sl::float2 &pt)
{
    return std::isfinite(pt.x) && std::isfinite(pt.y);
}

static inline bool valid3D(const sl::float3 &pt)
{
    return std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z);
}

static inline float dist3D(const sl::float3 &a, const sl::float3 &b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static inline int selectBestBodyIndex(const sl::Bodies &bodies)
{
    if (bodies.body_list.empty()) {
        return -1;
    }

    int best_idx = 0;
    int best_conf = bodies.body_list[0].confidence;

    for (int i = 1; i < static_cast<int>(bodies.body_list.size()); ++i) {
        if (bodies.body_list[i].confidence > best_conf) {
            best_conf = bodies.body_list[i].confidence;
            best_idx = i;
        }
    }

    return best_idx;
}

static bool hasArg(int argc, char **argv, const std::string &flag)
{
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

static std::string getFirstNonFlagArg(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (!arg.empty() && arg[0] != '-') {
            return arg;
        }
    }
    return "";
}