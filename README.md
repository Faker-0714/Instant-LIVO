# Instant-LIVO

**Instant-LIVO: LiDAR-Inertial-Visual Odometry**

This repository contains the implementation of **Instant-LIVO**, a tightly-coupled LiDAR-Inertial-Visual Odometry system based on [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2).

Our paper is available on [arXiv](https://arxiv.org/abs/2611.27633).

## Abstract

LiDAR–Inertial–Visual Odometry (LIVO) is a fundamental technology for autonomous systems; however, deployment on resource-constrained platforms remains challenging due to physical limitations. This paper presents Instant-LIVO, a tightly coupled, high-speed, real-time LIVO system. The system introduces a compact geometric map based on spatial-hashed OctVox and accelerates local-neighborhood retrieval and plane fitting through cache-assisted heuristic staged K-nearest-neighbor searches. It also employs a bounded single-reference-frame visual map that adaptively updates the reference frame according to the tracking status, while using Schmidt-type uncertainty-aware photometric fusion to maintain the correlation between the reference pose and the current state. The system performs multisensor fusion through a sequential process consisting of IMU prediction, LiDAR geometric correction, and visual photometric correction. It was evaluated on 32 public benchmarks and custom datasets. While maintaining competitive accuracy on both x86 and ARM platforms, Instant-LIVO achieves average processing-speed improvements of approximately 2.3× and 2.1×, respectively, and reduces peak memory usage to approximately 1/20 and 1/13, respectively. The results demonstrate that the proposed method outperforms existing LIO and LIVO baselines in both accuracy and computational efficiency.

### Key Contributions

1. **Compact OctVox mapping**: Hash voxels retain up to eight representative child-subvoxel points per block. Cache-accelerated C-HKNN performs staged searches with distance-bound early termination; on-demand planes support point-to-plane registration and image-patch warping
2. **Single-reference-frame map and uncertainty-aware photometric fusion**: bounded reference-frame map is adaptively replaced according to tracking status. Schmidt photometric updates preserve reference-pose uncertainty and state correlation through IMU prediction, LiDAR correction, and visual fusion
3. **Extensive experimental evaluation**: The tightly coupled LIVO framework is comprehensively evaluated on public benchmark datasets and self-collected datasets

## Features

- **Voxel-Based Mapping**: Eight subvoxels per block with nearest-neighbor plane fitting
- **Single Visual Reference**: Image patch tracking using a single visual reference frame
- **RGB Colorization**: Colored point cloud generation using camera observations
- **Multiple LiDAR Configurations**: Launch files for Livox MID-360, Hesai, Ouster, and RoboSense sensors
- **ROS Bag Input**: ROS topic playback and direct reading of bag files

## Environment Requirements

Instant-LIVO environment is compatible with FAST-LIVO2 requirements.

### Ubuntu and ROS

- Ubuntu 20.04
- ROS Noetic: [ROS Installation](http://wiki.ros.org/ROS/Installation)
- C++17

### Dependencies

| Library | Version | Installation |
|---------|---------|--------------|
| PCL | >=1.8 | [PCL Installation](https://pointclouds.org/downloads/) |
| Eigen | >=3.3.4 | [Eigen Installation](https://eigen.tuxfamily.org/index.php?title=Main_Page) |
| OpenCV | >=4.2 | [OpenCV Installation](https://opencv.org/get-started/) |
| Boost |  | [Boost Installation](https://www.boost.org/) |

### Sophus

Sophus installation for the non-templated/double-only version:

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build && cmake ..
make
sudo make install
```

### Vikit

Vikit contains camera models, math and interpolation functions. It is included in this repository.

## Build

Build with catkin:

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
mkdir -p src/Instant-LIVO/Log/{result,pcd,image,Colmap/sparse/0}
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## Configuration

### Sensor Setup

1. **Camera Intrinsics**: Configure the camera model in the corresponding `config/camera_*.yaml` file
2. **Sensor Extrinsics**: Set LiDAR-to-IMU and LiDAR-to-camera transformations in the dataset configuration
3. **ROS Topics**: Update the LiDAR, IMU, and image topics in the dataset configuration

Example configuration from `config/HILTI22.yaml`:

```yaml
common:
  img_topic: "/alphasense/cam0/image_raw"
  lid_topic: "/hesai/pandar"
  imu_topic: "/alphasense/imu"

extrin_calib:
  extrinsic_T: [-0.001, -0.00855, 0.055]
  extrinsic_R: [0, -1, 0,
                -1, 0, 0,
                0, 0, -1]
  Rcl: [-0.999926, -0.00670802, 0.0101073,
        -0.0100912, -0.00242564, -0.999946,
         0.00673218, -0.999975, 0.00235777]
  Pcl: [-0.0549762, 0.0675401, -0.0520599]
```

### VIO Parameters

```yaml
vio:
  reference_max_age: 8       # Maximum age of the visual reference
  visual_voxel_size: 0.5     # Visual-feature grid size in meters
  max_iterations: 5
  outlier_threshold: 500
  img_point_cov: 1000
  patch_size: 8
  patch_pyrimid_level: 4
  exposure_estimate_en: true
```

### LIO and Map Parameters

```yaml
lio:
  max_iterations: 5
  dept_err: 0.02
  beam_err: 0.05
  variance_floor: 0.001
  min_eigen_value: 0.0001

map:
  resolution: 0.4            # Voxel block edge in meters
  capacity: 100000000        # Maximum number of voxel blocks
```

## Run

Launch Instant-LIVO and play the bag file in separate terminals.

### On a Custom Dataset

Configure `config/mid_360.yaml` and `config/camera_pinhole.yaml` for your sensor setup:

```bash
roslaunch instant_livo mapping_mid_360.launch
rosbag play YOUR_DOWNLOADED.bag
```

### On Hilti SLAM Challenge

```bash
roslaunch instant_livo mapping_hesaixt32_hilti22.launch
rosbag play YOUR_DOWNLOADED.bag
```

The Hilti 2022 and 2023 handheld configurations use `preprocess/hilti_en` to process one out of every four camera frames.

### On Newer College

```bash
roslaunch instant_livo mapping_ouster_newercollege_collection_1.launch
rosbag play YOUR_DOWNLOADED.bag
```

### On NTU VIRAL

```bash
roslaunch instant_livo mapping_ouster_ntu.launch
rosbag play YOUR_DOWNLOADED.bag
```

### Additional Launch Files

| Dataset | Launch file |
|---------|-------------|
| Hilti 2023 handheld | `mapping_hesaixt32_hilti23_handheld.launch` |
| Hilti 2023 robot | `mapping_robosense_hilti23_robot.launch` |
| Newer College collection 4 | `mapping_ouster_newercollege_collection_4.launch` |

### Direct Bag Input

Read a bag directly using the topics in the selected dataset configuration:

```bash
roslaunch instant_livo mapping_hesaixt32_hilti22.launch \
  bag_enable:=true bag_path:=/path/to/dataset.bag
```

## System Architecture

```
             LiDAR  |  IMU  |  Camera
                    |
                    v
          Temporal Synchronization
                    |
                    v
              IMU Propagation
                    |
          +---------+---------+
          |                   |
          v                   v
     LiDAR Update        Visual Update
  Point-to-Plane ESIKF  Photometric ESIKF
          |                   |
          v                   v
   OctVox/HKNN Map    Visual Feature Map
          |                   |
          +---------+---------+
                    |
                    v
        Odometry and RGB Point Cloud
```

## Acknowledgments

We sincerely thank the authors of [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) for their outstanding work and for making their code open source. This project builds upon their foundation in LiDAR-Inertial-Visual Odometry.

## License

This project is released under the GNU General Public License v2. Please refer to the [LICENSE](LICENSE) file for details.

## Contact

For questions and collaboration opportunities, please contact:

- **XinRui Wang**: T589658@outlook.com

## Related Projects

- [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) - Our baseline system
