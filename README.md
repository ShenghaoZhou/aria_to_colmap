# Convert Project Aria result to COLMAP format

## Goal
This script aims to convert Project Aria result to COLMAP format, including raw VRS file and SLAM results from its MPS service. It will extract the image (from Aria rgb camera), camera pose and visible point clouds. 

## Input & Output
It expects a `vrs_file` (path to vrs file) and `mps_data_dir` (path to mps result folder), 
and it will dump the COLMAP files to `output_dir`.  

It expects `mps_data_dir` has the following structure
```bash
mps_data_dir
└── slam
    ├── closed_loop_trajectory.csv (camera pose from MPS SLAM)
    └── semidense_points.csv.gz (point cloud)
```
The output folder `output_dir` has the following structure
```bash
output_dir
├── images
│   ├── 000000.jpg
│   └── 000001.jpg
└── sparse
    └── 0
        ├── cameras.bin
        ├── database.db
        ├── images.bin
        ├── points3D.bin
        └── points3D.ply
```

## Details
- By default, the image will be undistored to a fixed size (line 110).
- In the current code, only first 100 frames are extracted (line 62). 
- Also, the world coordinate is specified [here](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/coordinate_convention/3d_coordinate_frame_convention). It is Aria "device frame", where z axis is not pointing up. 
- By default, there is no color in the provided point cloud. In the code, it will project the pixel color to the point clould. If there are more than one pixels for the point cloud, the last processed color will be used, as we find this seems to be more stable than the result of averaging over all visible pixels.



