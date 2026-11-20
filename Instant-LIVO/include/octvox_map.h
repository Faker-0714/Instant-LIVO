/* 
This file is part of Instant-LIVO

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef INSTANT_LIVO_OCTVOX_MAP_H
#define INSTANT_LIVO_OCTVOX_MAP_H

#include "common_lib.h"
#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>
#include <omp.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

struct PlaneEstimate
{
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Vector3d tangent = Eigen::Vector3d::Zero();
  Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
  double radius = 0.0;
  bool valid = false;

  double signedDistance(const Eigen::Vector3d &point) const;
  bool withinSupport(const Eigen::Vector3d &point) const;
};

// Eight bounded subvoxel means per block. Insertions own the map; queries are
// read-only and may run in parallel between insertion phases.
class OctVoxMap
{
  struct Voxel;
public:
  struct Options
  {
    double resolution = 0.5;
    std::size_t capacity = 100000000;
    double min_eigen_value = 0.0025;
  };

  // One cache per map/query worker; insertions invalidate cached addresses.
  struct QueryCache
  {
    Eigen::Vector3i fine = Eigen::Vector3i::Zero();
    uint64_t epoch = ~uint64_t(0), seen_blocks = 0;
    uint8_t groups = 0;
    std::array<uint16_t, 7> ends{};
    std::array<const Voxel *, 60> blocks{};
    std::vector<const Eigen::Vector3d *> candidates;
  };

  explicit OctVoxMap(const Options &options);
  OctVoxMap(const OctVoxMap &) = delete;
  OctVoxMap &operator=(const OctVoxMap &) = delete;

  void insert(const Eigen::Vector3d &point);
  PlaneEstimate queryPlane(const Eigen::Vector3d &point) const;
  PlaneEstimate queryPlane(const Eigen::Vector3d &point, QueryCache &cache) const;
  void visitBlocks(const std::function<void(const Eigen::Vector3d &)> &visitor) const;
  std::size_t size() const noexcept { return blocks_.size(); }

private:
  using Key = Eigen::Vector3i;
  struct KeyHash
  {
    std::size_t operator()(const Key &key) const noexcept;
  };
  struct Voxel
  {
    std::array<uint8_t, 8> counts{};
    std::array<Eigen::Vector3d, 8> means;
    void add(const Eigen::Vector3d &point, uint8_t index);
  };
  struct Neighbors;
  using Blocks = std::list<std::pair<Key, Voxel>>;

  bool validPoint(const Eigen::Vector3d &point) const;
  void nearest(const Eigen::Vector3d &point, Neighbors &neighbors, QueryCache &cache) const;

  Options options_;
  float resolution_;
  float sub_inverse_resolution_;
  std::array<double, 6> remaining_distance2_;
  Blocks blocks_;
  std::unordered_map<Key, Blocks::iterator, KeyHash> index_;
  uint64_t epoch_ = 0;
};

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

// LiDAR update configuration and map ownership.
struct VoxelMapConfig
{
  OctVoxMap::Options map;
  int max_iterations_ = 5;
  double beam_err_ = 0.02;
  double dept_err_ = 0.05;
  double sigma_num_ = 3.0;
  bool is_pub_plane_map_ = false;
  double variance_floor_ = 0.001;
};

struct PointToPlane
{
  Eigen::Vector3d point_w_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
  V3D rotation_jacobian_ = V3D::Zero();
  double inverse_variance_ = 0.0;
  double dis_to_plane_ = 0.0;
};

void calcBodyCov(Eigen::Vector3d &pb, float range_inc, float degree_inc, Eigen::Matrix3d &cov);
void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &config);

class VoxelMapManager
{
public:
  explicit VoxelMapManager(const VoxelMapConfig &config)
      : config_setting_(config), geometry_map_(config.map)
  {
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
  }

  const OctVoxMap &geometry() const noexcept { return geometry_map_; }
  void StateEstimation(StatesGroup &state_propagat);
  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);
  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list,
                            bool conservative_matching = false);
  void pubVoxelMap();

  VoxelMapConfig config_setting_;
  ros::Publisher voxel_map_pub_;
  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  M3D extR_ = M3D::Identity();
  V3D extT_ = V3D::Zero();
  StatesGroup state_;
  V3D position_last_ = V3D::Zero();
  geometry_msgs::Quaternion geoQuat_;
  int feats_down_size_ = 0;
  int effct_feat_num_ = 0;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

private:
  void pubSinglePlane(visualization_msgs::MarkerArray &markers, const PlaneEstimate &plane, int id);
  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);

  OctVoxMap geometry_map_;
  std::vector<PointToPlane> residual_buffer_;
  std::vector<uint8_t> residual_valid_;
  std::vector<OctVoxMap::QueryCache> search_caches_;
};
using VoxelMapManagerPtr = std::shared_ptr<VoxelMapManager>;

#endif // INSTANT_LIVO_OCTVOX_MAP_H
