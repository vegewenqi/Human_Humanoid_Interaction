import cv2
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

zed = sl.Camera()

record_video = False

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 60
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    exit(1)

positional_tracking_parameters = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(positional_tracking_parameters)

body_param = sl.BodyTrackingParameters()
body_param.enable_tracking = True
body_param.enable_body_fitting = True
body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
body_param.body_format = sl.BODY_FORMAT.BODY_38
zed.enable_body_tracking(body_param)

body_runtime_param = sl.BodyTrackingRuntimeParameters()
body_runtime_param.detection_confidence_threshold = 40

camera_info = zed.get_camera_information()
display_resolution = sl.Resolution(
    min(camera_info.camera_configuration.resolution.width, 1920),
    min(camera_info.camera_configuration.resolution.height, 1080)
)
image_scale = [
    display_resolution.width / camera_info.camera_configuration.resolution.width,
    display_resolution.height / camera_info.camera_configuration.resolution.height
]

viewer = gl.GLViewer()
viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam,
            body_param.enable_tracking, body_param.body_format)

bodies = sl.Bodies()
image = sl.Mat()
key_wait = 1

plot_init = False
fig = None

frames_2D = []
frames_3D = []
timestamps = []

previous_time = time.time()
while viewer.is_available():
    if zed.grab() <= sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT,
                           sl.MEM.BOTH, display_resolution)
        zed.retrieve_bodies(bodies, body_runtime_param)
        viewer.update_view(image, bodies)

        image_left_ocv = image.get_data()
        cv_viewer.render_2D(image_left_ocv, image_scale, bodies.body_list,
                            body_param.enable_tracking, body_param.body_format)

        if len(bodies.body_list) >= 2:

            bodyA = bodies.body_list[0]
            bodyB = bodies.body_list[1]

            joint_indices = [10, 11, 12, 13, 14, 15, 16, 17]

            for idx in joint_indices:

                JA_2D = bodyA.keypoint_2d[idx]
                JB_2D = bodyB.keypoint_2d[idx]

                if JA_2D[0] < 0 or JA_2D[1] < 0:
                    continue
                if JB_2D[0] < 0 or JB_2D[1] < 0:
                    continue

                JA_3D = bodyA.keypoint[idx]
                JB_3D = bodyB.keypoint[idx]

                if np.all(np.isfinite(JA_3D)) and np.all(np.isfinite(JB_3D)):
                    dist = np.linalg.norm(JA_3D - JB_3D)
                else:
                    dist = None

                JA = (int(JA_2D[0]), int(JA_2D[1]))
                JB = (int(JB_2D[0]), int(JB_2D[1]))

                cv2.line(image_left_ocv, JA, JB, (0, 255, 0), 2)
                cv2.circle(image_left_ocv, JA, 4, (0, 0, 255), -1)
                cv2.circle(image_left_ocv, JB, 4, (255, 0, 0), -1)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(image_left_ocv, f"{int(fps)} FPS",
                    (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("2D View", image_left_ocv)
        frame_bgr = cv2.cvtColor(image_left_ocv, cv2.COLOR_BGRA2BGR)

        if record_video == True:
            frames_2D.append(frame_bgr)
            timestamps.append(time.time())

            frame_points = []
            for body in bodies.body_list:
                joints = [joint.tolist() for joint in body.keypoint]
                frame_points.append(joints)
            frames_3D.append(frame_points)

        # 3D plot
        # Comment out this block of code if you need maximum performance
        # if plot_init == False:
        #     plt.ion()
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.set_xlabel('X [m]')
        #     ax.set_ylabel('Y [m]')
        #     ax.set_zlabel('Z [m]')
        #     ax.set_xlim([-2, 2])
        #     ax.set_ylim([-2, 0])
        #     ax.set_zlim([-2, 0])
        #     plot_init = True

        # ax.cla()
        # ax.set_xlabel('X [m]')
        # ax.set_ylabel('Y [m]')
        # ax.set_zlabel('Z [m]')
        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 0])
        # ax.set_zlim([-2, 0])

        # for body in bodies.body_list:
        #     xs, ys, zs = [], [], []
        #     for joint in body.keypoint:
        #         xs.append(joint[0])
        #         ys.append(joint[2])
        #         zs.append(joint[1])
        #     ax.scatter(xs, ys, zs)

        # plt.draw()
        # plt.pause(0.001)

        key = cv2.waitKey(key_wait)
        if key == ord('q'):
            break
        if key == ord('p'):
            key_wait = 0 if key_wait > 0 else 10

if record_video == True:
    durations = np.diff(timestamps)
    avg_fps = 1.0 / np.mean(durations)
    if len(frames_2D) > 0:
        h, w = frames_2D[0].shape[:2]
        video_out = cv2.VideoWriter(
            "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), avg_fps, (w, h))
        for f in frames_2D:
            video_out.write(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(6, 6)

    plot_video_out = None

    for frame_points in frames_3D:
        ax.cla()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 0])
        ax.set_zlim([-2, 0])
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        for body in frame_points:
            xs, ys, zs = zip(*[(j[0], j[2], j[1]) for j in body])
            ax.scatter(xs, ys, zs)

        fig.canvas.draw()
        frame_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame_argb = frame_argb.reshape(
            fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame_argb[..., [1, 2, 3]]
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if plot_video_out is None:
            h, w = frame_bgr.shape[:2]
            plot_video_out = cv2.VideoWriter(
                "3D_plot.mp4", cv2.VideoWriter_fourcc(*"mp4v"), avg_fps, (w, h))

        plot_video_out.write(frame_bgr)
        plot_video_out.release()
        plt.close(fig)
        video_out.release()

# Close everything
print("Closing...")
viewer.exit()
image.free(sl.MEM.CPU)
zed.disable_body_tracking()
zed.disable_positional_tracking()
zed.close()
cv2.destroyAllWindows()
