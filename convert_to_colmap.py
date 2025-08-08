"""
Extract frames from vrs, read camera intrinsics and extrinsics from mps (Meta internal SLAM system),
and read out 3D points from mps point cloud. The processing logic is adapted from nerfstudio & project aria demo

In this way, we can use COLMAP GUI to visualize the results
"""
# watch out for potentiailly different world frame conventions
# ref: https://github.com/facebookresearch/projectaria_tools/issues/157 (no it is not applicable)

from pathlib import Path
import shutil
import numpy as np
from projectaria_tools.core.data_provider import create_vrs_data_provider
# from projectaria_tools.core.image import InterpolationMethod
# from projectaria_tools.core import calibration
import projectaria_tools.core as aria_core
from projectaria_tools.core.mps.utils import get_nearest_pose, filter_points_from_confidence
import pycolmap
from PIL import Image
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from tqdm import tqdm
import pandas as pd
# To establish the mapping between mps pose and vrs recording,
# from mps to vrs: provider.get_image_data_by_time_ns
# from vrs to mps: projectaria_tools.core.mps.utils.get_nearest_pose (binary search)


def get_pcd_from_mps(mps_data_dir):
    points_path = mps_data_dir / "global_points.csv.gz"
    if not points_path.exists():
        # MPS point cloud output was renamed in Aria's December 4th, 2023 update.
        # https://facebookresearch.github.io/projectaria_tools/docs/ARK/sw_release_notes#project-aria-updates-aria-mobile-app-v140-and-changes-to-mps
        points_path = mps_data_dir / "semidense_points.csv.gz"

    # read point cloud
    points_data = aria_core.mps.read_global_point_cloud(str(points_path))
    points_data = filter_points_from_confidence(
        points_data)
    points_uid = np.array([point.uid for point in points_data])
    points_xyz = np.array([point.position_world for point in points_data])
    return points_xyz, points_uid


def get_posed_images(provider, mps_traj, stream_id, start_idx=0, num_data=100):
    """
    Generator that yields (image_data, extrinsics) tuples for each image in the provider,
    where extrinsics is a 4x4 matrix from get_nearest_pose.
    """
    # vrs to mps
    max_num_data = provider.get_num_data(stream_id)
    print(f"Total {max_num_data} images in stream {stream_id}.")
    print(f"Start from index {start_idx}, total {num_data} images to process.")
    cnt = 0
    for idx in range(start_idx, max_num_data):

        image_data = provider.get_image_data_by_index(
            stream_id, idx)
        pose = get_nearest_pose(
            mps_traj, image_data[1].capture_timestamp_ns)
        if pose is None:
            continue
        else:
            cnt += 1
            if cnt > num_data:
                break
        # this is Pose of the device coordinate frame in world frame
        pose_mat = pose.transform_world_device.to_matrix()
        # there is extra device to rgb camera transformation

        # extrinsics is the inverse of pose, but we want to avoid the matrix inversion
        # as it has special close-form solution

        # FIXME: currently, there seems to be a constant offset. I suppose it is caused by the difference between device coordinate and the rgb camera, which we don't account for
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = pose_mat[:3, :3].T
        extrinsics[:3, 3] = -pose_mat[:3, :3].T @ pose_mat[:3, 3]
        yield image_data, extrinsics


def get_pcd_visibility_from_mps(mps_data_dir):
    observations_path = mps_data_dir / "semidense_observations.csv.gz"
    # WARN: this is known to be slow
    print(f"Reading MPS point cloud visibility from {observations_path}...")
    print("This may take a while...")
    observations = aria_core.mps.read_point_observations(
        str(observations_path))
    print(f"Loaded {len(observations)} observations from MPS.")
    # observations is a list of objects with attributes
    # we convert it to a dataframe with the same attributes for easier querying
    # columns = [name for name in dir(
    #     observations[0]) if not name.startswith('_')]
    columns = ['frame_capture_timestamp', 'point_uid']
    obs_dict = {col: [getattr(obs, col)
                      for obs in observations] for col in columns}
    # obs_dict['u'] = [obs.uv[0] for obs in observations]
    # obs_dict['v'] = [obs.uv[1] for obs in observations]
    obs_df = pd.DataFrame.from_dict(obs_dict)
    print(f"Finished loading {len(obs_df)} observations in Pandas")
    return obs_df


def find_visible_pcd_this_frame(pcd_uid, pcd_visibility, capture_timestamp_ns):

    idx = np.searchsorted(
        pcd_visibility['frame_capture_timestamp'], capture_timestamp_ns)
    closest_timestamp = pcd_visibility.iloc[idx].frame_capture_timestamp
    selected = pcd_visibility['frame_capture_timestamp'] == closest_timestamp
    pcd_visible_uid = pcd_visibility[selected]["point_uid"]
    pcd_visible_idx = np.where(np.isin(pcd_uid, pcd_visible_uid))[0]
    return pcd_visible_idx


def test_pts_in_cam(pcd, cam_from_world, colmap_cam, tgt_img_size):
    rot = cam_from_world.rotation.matrix()
    trans = cam_from_world.translation
    pcd_cam = np.einsum('ij, kj->ki', rot, pcd) + trans
    pcd_2d = colmap_cam.img_from_cam(pcd_cam)

    pcd_is_visible_this_frame = (pcd_2d[:, 0] >= 0) & (pcd_2d[:, 0] < tgt_img_size) & \
        (pcd_2d[:, 1] >= 0) & (pcd_2d[:, 1] < tgt_img_size)

    # only consider points in front of the camera
    pcd_is_visible_this_frame &= pcd_cam[:, 2] > 0

    pcd_2d_xy = pcd_2d[pcd_is_visible_this_frame].astype(np.int32)
    return pcd_is_visible_this_frame, pcd_2d_xy


def main(vrs_file: Path, mps_data_dir: Path, output_dir: Path, start_idx=0, num_data=100, force_delete=True):
    if force_delete and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = create_vrs_data_provider(str(vrs_file.absolute()))
    assert provider is not None, "Cannot open file"

    mps_traj_file = mps_data_dir / "closed_loop_trajectory.csv"
    mps_traj = aria_core.mps.read_closed_loop_trajectory(str(mps_traj_file))

    # turn on color correction to fix overly blue bug
    provider.set_color_correction(True)
    provider.set_devignetting(False)

    # focus on RGB camera images in the VRS
    sensor_name = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(sensor_name)

    # get camera calibration for intrinsics
    rgb_calib = provider.get_device_calibration().get_camera_calib(sensor_name)

    # set our target undistorted intrinsics
    tgt_img_size = 480
    tgt_focal = 250.0  # focal length in pixels
    dst_calib = aria_core.calibration.get_linear_camera_calibration(
        tgt_img_size, tgt_img_size, tgt_focal, sensor_name + "-undistorted")

    rec = pycolmap.Reconstruction()
    # TODO: add as COLMAP camera
    colmap_cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=tgt_img_size, height=tgt_img_size,
        params=np.array([tgt_focal, tgt_img_size / 2, tgt_img_size / 2]),
        camera_id=0,
    )
    rec.add_camera(colmap_cam)

    # Directly adding 3D points from MPS to COLMAP is not ideal:
    # First, we have too many points, as it is for the whole scene, but only a subset are visible in the images.
    # Second, there is no color on points, which is a problem for 3DGS initialization
    # TODO: after camera pos is added, we add 3D points, and figure out the visibility by projection,
    # and take the color as the mean color of visible pixels
    pcd, pcd_uid = get_pcd_from_mps(mps_data_dir)

    # use official visibility from MPS. WARN: this is known to be slow
    pcd_visibility = get_pcd_visibility_from_mps(mps_data_dir)

    pcd_is_visible = np.zeros(len(pcd), dtype=bool)
    pcd_colors = np.zeros((len(pcd), 3), dtype=np.uint8)

    img_save_dir = output_dir / "images"
    img_save_dir.mkdir(exist_ok=True)
    img_names = []

    # camera_serial = rgb_calib.get_serial_number()

    for idx, (image_data, extrinsics_device) in tqdm(enumerate(get_posed_images(provider, mps_traj, stream_id,
                                                                                start_idx, num_data)),):
        # undistort the image
        rectified_array = aria_core.calibration.distort_by_calibration(
            image_data[0].to_numpy_array(),
            dst_calib, rgb_calib,
            aria_core.image.InterpolationMethod.BILINEAR)

        # extrinsics w.r.t device, need to further transform to current camera
        extrinsics = np.linalg.inv(
            rgb_calib.get_transform_device_camera().to_matrix()) @ extrinsics_device

        # save the image with patterned name, so it can match the entry in database
        # img_name = f"{image_data[1].capture_timestamp_ns}.jpg"
        img_name = f"{idx:06d}.jpg"
        img_path = img_save_dir / img_name
        Image.fromarray(rectified_array).save(img_path)
        img_names.append(img_name)

        # TODO: add as COLMAP image
        # notice: COLMAP image focuses on the "camera pose + 2D keypoints", not raw image data,
        # as it assumes the image has been processed by feature extractor
        colmap_im = pycolmap.Image(
            id=idx,
            name=img_name,
            camera_id=colmap_cam.camera_id,
            cam_from_world=pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(extrinsics[:3, :3]),
                translation=extrinsics[:3, 3]
            )
        )
        colmap_im.registered = True

        # Retrieve visibility information from MPS result
        # pcd_2d is visible keypoint uv coordinate on the image,
        # it corresponds to the pcd_visible_idx in all pointclouds
        pcd_potential_visible_idx = find_visible_pcd_this_frame(
            pcd_uid, pcd_visibility, image_data[1].capture_timestamp_ns)

        pcd_potential_visible = pcd[pcd_potential_visible_idx]

        pcd_potential_visible_is_visible, pcd_2d_xy = test_pts_in_cam(
            pcd_potential_visible, colmap_im.cam_from_world, colmap_cam, tgt_img_size)

        pcd_is_visible_this_frame = np.zeros(len(pcd), dtype=bool)
        pcd_is_visible_this_frame[pcd_potential_visible_idx[pcd_potential_visible_is_visible]] = True

        pcd_is_visible |= pcd_is_visible_this_frame
        pcd_colors[pcd_is_visible_this_frame] = rectified_array[pcd_2d_xy[:, 1], pcd_2d_xy[:, 0]]

        # added the most updated visibility relation
        colmap_im.points2D = pycolmap.ListPoint2D(
            [pycolmap.Point2D(p, id_) for p, id_ in zip(
                pcd_2d_xy, np.nonzero(pcd_is_visible_this_frame)[0])],
        )
        rec.add_image(colmap_im)

    # collect visible points, and add them to COLMAP
    print('adding visible points to COLMAP...')
    pcd_visible = pcd[pcd_is_visible]
    pcd_colors_visible = pcd_colors[pcd_is_visible].clip(
        0, 255).astype(np.uint8)
    print(f'Found {len(pcd_visible)} visible points in total.')
    for point, color in zip(pcd_visible, pcd_colors_visible):
        rec.add_point3D(point, pycolmap.Track(), color)

    output_model_dir = output_dir / "sparse/0"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_model_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    database_path.touch()
    # create COLMAP database
    database = pycolmap.Database(str(database_path))
    database.write_camera(colmap_cam, use_camera_id=True)
    database.close()
    pycolmap.import_images(
        database_path=str(database_path),
        image_path=str(img_save_dir),
        image_list=img_names,
        options=pycolmap.ImageReaderOptions(
            existing_camera_id=0
        )
    )

    # optional: we can add 2D observations of these points as well

    # save to COLMAP results
    rec.write(str(output_model_dir))


if __name__ == "__main__":
    import tyro
    tyro.cli(main, description="Convert MPS data to COLMAP format for visualization.")
