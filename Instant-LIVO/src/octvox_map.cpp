/*
This file is part of Instant-LIVO

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "octvox_map.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
// Six search groups, mirrored for each of the eight subvoxel parities.
alignas(16) static const std::array<Eigen::Vector3i, 60> neighbor_blocks =
{
  Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(0, -1, 0), Eigen::Vector3i(0, 0, -1), Eigen::Vector3i(-1, 0, 0), Eigen::Vector3i(-1, 0, -1), Eigen::Vector3i(0, -1, -1),
  Eigen::Vector3i(-1, -1, 0), Eigen::Vector3i(-1, -1, -1), Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(0, 0, 1), Eigen::Vector3i(0, -1, 1),
  Eigen::Vector3i(1, -1, 0), Eigen::Vector3i(1, 0, -1), Eigen::Vector3i(0, 1, -1), Eigen::Vector3i(-1, 1, 0), Eigen::Vector3i(-1, 0, 1), Eigen::Vector3i(-1, -1, 1),
  Eigen::Vector3i(-1, 1, -1), Eigen::Vector3i(1, -1, -1), Eigen::Vector3i(0, 0, -2), Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 0), Eigen::Vector3i(1, 0, 1),
  Eigen::Vector3i(-2, 0, 0), Eigen::Vector3i(0, -2, 0), Eigen::Vector3i(-2, 0, -1), Eigen::Vector3i(-1, 0, -2), Eigen::Vector3i(-1, -2, 0), Eigen::Vector3i(-2, -1, 0),
  Eigen::Vector3i(0, -2, -1), Eigen::Vector3i(1, 1, -1), Eigen::Vector3i(-1, 1, 1), Eigen::Vector3i(1, -1, 1), Eigen::Vector3i(0, -1, -2), Eigen::Vector3i(-2, -1, -1),
  Eigen::Vector3i(-1, -1, -2), Eigen::Vector3i(-1, -2, -1), Eigen::Vector3i(-2, 1, 0), Eigen::Vector3i(1, -2, 0), Eigen::Vector3i(0, -2, 1), Eigen::Vector3i(0, 1, -2),
  Eigen::Vector3i(1, 0, -2), Eigen::Vector3i(1, 1, 1), Eigen::Vector3i(-2, 0, 1), Eigen::Vector3i(1, -1, -2), Eigen::Vector3i(1, -2, -1), Eigen::Vector3i(-2, 1, -1),
  Eigen::Vector3i(-2, -1, 1), Eigen::Vector3i(-1, 1, -2), Eigen::Vector3i(-1, -2, 1), Eigen::Vector3i(0, -2, -2), Eigen::Vector3i(-2, 0, -2), Eigen::Vector3i(-2, 1, 1),
  Eigen::Vector3i(1, 1, -2), Eigen::Vector3i(1, -2, 1), Eigen::Vector3i(-2, -2, 0), Eigen::Vector3i(-2, -1, -2), Eigen::Vector3i(-2, -2, -1), Eigen::Vector3i(-1, -2, -2),
  };

static constexpr std::array<uint16_t, 7> search_offsets = {{
  0, 43, 135, 219, 321, 465, 593,
}};

static constexpr std::array<uint8_t, 593> search_order = {{
  // Group0
  0, 8, 0, 1, 2, 3, 4, 5, 6, 7,
  1, 4, 2, 3, 6, 7,
  2, 4, 4, 5, 6, 7,
  3, 4, 1, 3, 5, 7,
  4, 2, 5, 7,
  5, 2, 6, 7,
  6, 2, 3, 7,
  7, 1, 7,

  // Group1
  1, 4, 0, 1, 4, 5,
  2, 4, 0, 1, 2, 3,
  3, 4, 0, 2, 4, 6,
  4, 4, 1, 3, 4, 6,
  5, 4, 2, 3, 4, 5,
  6, 4, 1, 2, 5, 6,
  7, 3, 3, 5, 6,
  8, 4, 0, 2, 4, 6,
  9, 4, 0, 1, 4, 5,
  10, 4, 0, 1, 2, 3,
  11, 2, 2, 3,
  12, 2, 2, 6,
  13, 2, 4, 6,
  14, 2, 4, 5,
  15, 2, 1, 5,
  16, 2, 1, 3,
  17, 1, 3,
  18, 1, 5,
  19, 1, 6,

  // Group2
  4, 2, 0, 2,
  5, 2, 0, 1,
  6, 2, 0, 4,
  7, 4, 0, 1, 2, 4,
  11, 2, 0, 1,
  12, 2, 0, 4,
  13, 2, 0, 2,
  14, 2, 0, 1,
  15, 2, 0, 4,
  16, 2, 0, 2,
  17, 3, 0, 1, 2,
  18, 3, 0, 1, 4,
  19, 3, 0, 2, 4,
  21, 2, 0, 1,
  22, 2, 0, 4,
  23, 2, 0, 2,
  31, 2, 0, 4,
  32, 2, 0, 1,
  33, 2, 0, 2,
  43, 1, 0,

  // Group3
  8, 4, 1, 3, 5, 7,
  9, 4, 2, 3, 6, 7,
  10, 4, 4, 5, 6, 7,
  11, 2, 6, 7,
  12, 2, 3, 7,
  13, 2, 5, 7,
  14, 2, 6, 7,
  15, 2, 3, 7,
  16, 2, 5, 7,
  17, 1, 7,
  18, 1, 7,
  19, 1, 7,
  20, 4, 4, 5, 6, 7,
  24, 4, 1, 3, 5, 7,
  25, 4, 2, 3, 6, 7,
  26, 2, 5, 7,
  27, 2, 5, 7,
  28, 2, 3, 7,
  29, 2, 3, 7,
  30, 2, 6, 7,
  34, 2, 6, 7,
  35, 1, 7,
  36, 1, 7,
  37, 1, 7,

  // Group4
  11, 2, 4, 5,
  12, 2, 1, 5,
  13, 2, 1, 3,
  14, 2, 2, 3,
  15, 2, 2, 6,
  16, 2, 4, 6,
  17, 2, 5, 6,
  18, 2, 3, 6,
  19, 2, 3, 5,
  21, 4, 2, 3, 4, 5,
  22, 4, 1, 2, 5, 6,
  23, 4, 1, 3, 4, 6,
  26, 2, 1, 3,
  27, 2, 4, 6,
  28, 2, 2, 6,
  29, 2, 1, 5,
  30, 2, 2, 3,
  31, 2, 5, 6,
  32, 2, 3, 5,
  33, 2, 3, 6,
  34, 2, 4, 5,
  35, 2, 3, 5,
  36, 2, 5, 6,
  37, 2, 3, 6,
  38, 2, 1, 5,
  39, 2, 2, 6,
  40, 2, 2, 3,
  41, 2, 4, 5,
  42, 2, 4, 6,
  44, 2, 1, 3,
  45, 1, 6,
  46, 1, 6,
  47, 1, 5,
  48, 1, 3,
  49, 1, 5,
  50, 1, 3,

  // Group5
  17, 1, 4,
  18, 1, 2,
  19, 1, 1,
  21, 2, 6, 7,
  22, 2, 3, 7,
  23, 2, 5, 7,
  31, 3, 1, 2, 7,
  32, 3, 2, 4, 7,
  33, 3, 1, 4, 7,
  35, 1, 1,
  36, 1, 4,
  37, 1, 2,
  38, 2, 3, 7,
  39, 2, 3, 7,
  40, 2, 6, 7,
  41, 2, 6, 7,
  42, 2, 5, 7,
  43, 3, 1, 2, 4,
  44, 2, 5, 7,
  45, 2, 4, 7,
  46, 2, 2, 7,
  47, 2, 1, 7,
  48, 2, 1, 7,
  49, 2, 4, 7,
  50, 2, 2, 7,
  51, 2, 6, 7,
  52, 2, 5, 7,
  53, 1, 1,
  54, 1, 4,
  55, 1, 2,
  56, 2, 3, 7,
  57, 1, 7,
  58, 1, 7,
  59, 1, 7,

}};

} // namespace

struct OctVoxMap::Neighbors
{
  int count = 0;
  int worst = 0;
  float max_distance2 = 0.0f;
  std::array<float, 5> distances{};
  std::array<Eigen::Vector3d, 5> points;

  void add(float distance2, const Eigen::Vector3d &point)
  {
    if (count == 5 && distance2 >= max_distance2) return;
    const int slot = count < 5 ? count : worst;
    distances[slot] = distance2;
    points[slot] = point;
    if (count < 5)
    {
      ++count;
      if (distance2 > max_distance2)
      {
        max_distance2 = distance2;
        worst = slot;
      }
    }
    else
    {
      // Equal maxima choose the last slot.
      worst = 0;
      for (int i = 1; i < 5; ++i)
        if (distances[i] >= distances[worst]) worst = i;
      max_distance2 = distances[worst];
    }
  }
};

void OctVoxMap::Voxel::add(const Eigen::Vector3d &point, uint8_t index)
{
  auto &count = counts[index];
  auto &mean = means[index];
  if (count == 0)
  {
    mean = point;
    count = 1;
  }
  else if (count < 20 && (point - mean).squaredNorm() <= 0.1 * 0.1)
  {
    mean += (point - mean) / static_cast<double>(++count);
  }
}

std::size_t OctVoxMap::KeyHash::operator()(const Key &key) const noexcept
{
  std::size_t hash = static_cast<std::size_t>(key.x());
  hash ^= key.y() * 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= key.z() * 0x85ebca6b + (hash << 6) + (hash >> 2);
  return hash;
}

OctVoxMap::OctVoxMap(const Options &options)
    : options_(options), resolution_(options.resolution)
{
  if (!std::isfinite(resolution_) || resolution_ <= 0.0f || options.capacity == 0 ||
      !std::isfinite(options.min_eigen_value) || options.min_eigen_value <= 0.0)
    throw std::invalid_argument("OctVox resolution, capacity and plane threshold must be positive");
  const float sub_resolution = resolution_ / 2.0f;
  sub_inverse_resolution_ = 1.0f / sub_resolution;

  // Bound every unvisited subcell after each group, in metres squared.
  double remaining = std::numeric_limits<double>::infinity();
  const double scale2 = static_cast<double>(sub_resolution) * sub_resolution;
  for (std::size_t group = remaining_distance2_.size(); group-- > 0;)
  {
    remaining_distance2_[group] = scale2 * remaining;
    std::size_t offset = search_offsets[group];
    while (offset < search_offsets[group + 1])
    {
      const auto &neighbor = neighbor_blocks[search_order[offset++]];
      const uint8_t count = search_order[offset++];
      for (uint8_t item = 0; item < count; ++item)
      {
        const uint8_t local = search_order[offset++];
        double distance2 = 0.0;
        for (int axis = 0; axis < 3; ++axis)
        {
          const int delta = 2 * neighbor[axis] + ((local >> axis) & 1);
          const int gap = std::max(0, std::max(-delta - 1, delta - 1));
          distance2 += gap * gap;
        }
        remaining = std::min(remaining, distance2);
      }
    }
  }
}

bool OctVoxMap::validPoint(const Eigen::Vector3d &point) const
{
  // Keep fine-cell conversion and all stencil offsets inside the integer range.
  return point.allFinite() &&
      (point.cwiseAbs().array() * (2.0 / options_.resolution) < 2147483000.0).all();
}

void OctVoxMap::insert(const Eigen::Vector3d &point)
{
  if (!validPoint(point)) return;
  ++epoch_;
  const Key fine = (point * sub_inverse_resolution_).array().floor().cast<int>();
  const Key key(fine.x() >> 1, fine.y() >> 1, fine.z() >> 1);
  const uint8_t local = ((fine.z() & 1) << 2) | ((fine.y() & 1) << 1) | (fine.x() & 1);
  auto found = index_.find(key);
  if (found == index_.end())
  {
    blocks_.emplace_front(key, Voxel{});
    blocks_.front().second.add(point, local);
    index_.emplace(key, blocks_.begin());
    if (blocks_.size() > options_.capacity)
    {
      index_.erase(blocks_.back().first);
      blocks_.pop_back();
    }
  }
  else
  {
    found->second->second.add(point, local);
    blocks_.splice(blocks_.begin(), blocks_, found->second);
  }
}

void OctVoxMap::nearest(const Eigen::Vector3d &point, Neighbors &neighbors, QueryCache &cache) const
{
  const Key fine = (point * sub_inverse_resolution_).array().floor().cast<int>();
  if (cache.epoch != epoch_ || cache.fine != fine)
  {
    cache.epoch = epoch_;
    cache.fine = fine;
    cache.groups = 0;
    cache.seen_blocks = 0;
    cache.ends[0] = 0;
    cache.candidates.clear();
    if (cache.candidates.capacity() == 0) cache.candidates.reserve(64);
  }
  const Key key(fine.x() >> 1, fine.y() >> 1, fine.z() >> 1);
  const int dx = fine.x() & 1, dy = fine.y() & 1, dz = fine.z() & 1;
  const int local = (dz << 2) | (dy << 1) | dx;
  const Key mirror(1 - (dx << 1), 1 - (dy << 1), 1 - (dz << 1));
  for (std::size_t group = 0; group < remaining_distance2_.size(); ++group)
  {
    if (group < cache.groups)
    {
      for (uint16_t i = cache.ends[group]; i < cache.ends[group + 1]; ++i)
      {
        const auto &p = *cache.candidates[i];
        neighbors.add((p - point).squaredNorm(), p);
      }
    }
    else
    {
      const uint8_t *entry = search_order.data() + search_offsets[group];
      const uint8_t *end = search_order.data() + search_offsets[group + 1];
      while (entry < end)
      {
        const uint8_t block_index = *entry++;
        const uint8_t count = *entry++;
        const uint64_t bit = uint64_t(1) << block_index;
        if (!(cache.seen_blocks & bit))
        {
          const auto found = index_.find(key + mirror.cwiseProduct(neighbor_blocks[block_index]));
          cache.blocks[block_index] = found == index_.end() ? nullptr : &found->second->second;
          cache.seen_blocks |= bit;
        }
        const Voxel *voxel = cache.blocks[block_index];
        if (voxel)
          for (uint8_t i = 0; i < count; ++i)
          {
            const uint8_t index = entry[i] ^ local;
            if (!voxel->counts[index]) continue;
            const auto &p = voxel->means[index];
            cache.candidates.push_back(&p);
            neighbors.add((p - point).squaredNorm(), p);
          }
        entry += count;
      }
      cache.ends[group + 1] = static_cast<uint16_t>(cache.candidates.size());
      ++cache.groups;
    }
    // Identical groups, candidate order, float distances and stopping rule.
    if (neighbors.count == 5 && neighbors.max_distance2 < remaining_distance2_[group]) break;
  }
}

void OctVoxMap::visitBlocks(const std::function<void(const Eigen::Vector3d &)> &visitor) const
{
  for (const auto &block : blocks_)
    visitor((block.first.cast<double>().array() + 0.5).matrix() * resolution_);
}

double PlaneEstimate::signedDistance(const Eigen::Vector3d& point) const { return normal.dot(point - center); }
bool PlaneEstimate::withinSupport(const Eigen::Vector3d& point) const {
  // Evaluate the finite patch extent at each query.
  if (!valid || !point.allFinite()) return false;
  const Eigen::Vector3d delta = point - center;
  const double distance = normal.dot(delta);
  return std::max(0.0, delta.squaredNorm() - distance * distance) <= 9.0 * radius * radius;
}
PlaneEstimate OctVoxMap::queryPlane(const Eigen::Vector3d& point) const {
  QueryCache cache;
  return queryPlane(point, cache);
}

PlaneEstimate OctVoxMap::queryPlane(const Eigen::Vector3d& point, QueryCache &cache) const {
  if (!validPoint(point)) return {};
  Neighbors heap;
  nearest(point, heap, cache);
  const int count = heap.count;
  if (count < 4) return {};

  PlaneEstimate plane;
  for (int i = 0; i < count; ++i) plane.center += heap.points[i];
  plane.center /= count;
  Eigen::Matrix3d scatter = Eigen::Matrix3d::Zero();
  std::array<Eigen::Vector3d, 5> centered;
  for (int i = 0; i < count; ++i) {
    centered[i] = heap.points[i] - plane.center;
    scatter.noalias() += centered[i] * centered[i].transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) return {};
  const auto values = solver.eigenvalues();
  const double floor = std::numeric_limits<double>::epsilon() * std::max(1.0, values.maxCoeff());
  if (values(1) - values(0) <= floor) return {};
  plane.normal = solver.eigenvectors().col(0);
  plane.tangent = solver.eigenvectors().col(2);
  plane.eigenvalues = values / count;
  const double subvoxel = options_.resolution * 0.5;
  const double gap = 1e-6 * std::max(plane.eigenvalues[2], subvoxel * subvoxel);
  if (!(plane.eigenvalues[0] < options_.min_eigen_value &&
        plane.eigenvalues[1] - plane.eigenvalues[0] > gap &&
        plane.eigenvalues[2] - plane.eigenvalues[0] > gap)) return {};
  for (int i = 0; i < count; ++i)
    if (std::abs(plane.normal.dot(centered[i])) > 0.1) return {};
  plane.radius = std::sqrt(std::max(0.0, plane.eigenvalues[2]));
  plane.valid = true;
  return plane;
}

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &config)
{
  nh.param<double>("map/resolution", config.map.resolution, 0.5);
  nh.param<double>("lio/min_eigen_value", config.map.min_eigen_value, 0.0025);
  int capacity;
  nh.param<int>("map/capacity", capacity, 100000000);
  if (capacity <= 0) throw std::invalid_argument("map/capacity must be positive");
  config.map.capacity = static_cast<std::size_t>(capacity);
  nh.param<double>("lio/variance_floor", config.variance_floor_, 0.001);
  nh.param<int>("lio/max_iterations", config.max_iterations_, 5);
  nh.param<double>("lio/sigma_num", config.sigma_num_, 3.0);
  nh.param<double>("lio/beam_err", config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", config.dept_err_, 0.05);
  nh.param<bool>("publish/pub_plane_en", config.is_pub_plane_map_, false);
  if (config.max_iterations_ <= 0 || !std::isfinite(config.sigma_num_) || config.sigma_num_ <= 0.0 ||
      !std::isfinite(config.variance_floor_) || config.variance_floor_ <= 0.0)
    throw std::invalid_argument("Invalid LIO configuration");
}

void VoxelMapManager::StateEstimation(StatesGroup &state_propagat)
{
  pv_list_.resize(feats_down_size_);
#ifdef MP_EN
#pragma omp parallel for num_threads(std::min(MP_PROC_NUM, omp_get_max_threads())) schedule(static)
#endif
  for (int i = 0; i < feats_down_size_; ++i) {
    auto &point = pv_list_[i];
    const auto &raw = feats_down_body_->points[i];
    point.point_b << raw.x, raw.y, raw.z;
    point.point_i = extR_ * point.point_b + extT_;
    V3D covariance_point = point.point_b;
    if (covariance_point.z() == 0.0) covariance_point.z() = 0.001;
    calcBodyCov(covariance_point, config_setting_.dept_err_, config_setting_.beam_err_, point.body_var);
  }

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();
  // The prior covariance is fixed throughout this iterated update.
  const MD(DIM_STATE, DIM_STATE) prior_information = state_.cov.inverse();
  const StatesGroup predicted_state = state_;
  bool conservative_matching = false;
  // The map query is local. A correction spanning two blocks invalidates its
  // correspondences, even when a broad predicted covariance admitted them.
  const double max_correction = 2.0 * config_setting_.map.resolution;

  bool flg_EKF_converged, EKF_stop_flg = 0;
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    ptpl_list_.clear();

    BuildResidualListOMP(pv_list_, ptpl_list_, conservative_matching);

    effct_feat_num_ = ptpl_list_.size();
    if (effct_feat_num_ == 0)
    {
      position_last_ = state_.pos_end;
      break;
    }

    /*** Computation of Measuremnt Jacobian matrix H and measurents covarience
     * ***/
    MatrixXd Hsub(effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
    VectorXd R_inv(effct_feat_num_);
    VectorXd meas_vec(effct_feat_num_);
    meas_vec.setZero();
    for (int i = 0; i < effct_feat_num_; i++)
    {
      auto &ptpl = ptpl_list_[i];
      const V3D &A = ptpl.rotation_jacobian_;
      R_inv(i) = ptpl.inverse_variance_;
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), ptpl_list_[i].normal_[0] * R_inv(i),
          ptpl_list_[i].normal_[1] * R_inv(i), ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
    }
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    /*** Iterative Kalman Filter Update ***/
    auto &&HTz = Hsub_T_R_inv * meas_vec;
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + prior_information).inverse();
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    auto vec = state_propagat - state_;
    VD(DIM_STATE)
    solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

    const V3D corrected_position = state_.pos_end + solution.segment<3>(3);
    if (!solution.allFinite() || (corrected_position - predicted_state.pos_end).norm() > max_correction)
    {
      state_ = predicted_state;
      if (!conservative_matching)
      {
        ROS_WARN("LIO correction exceeded local map support; retrying with measurement-only association gate");
        conservative_matching = true;
        rematch_num = 0;
        iterCount = -1;
        continue;
      }
      ROS_WARN("LIO conservative update rejected; retaining IMU prediction");
      ptpl_list_.clear();
      effct_feat_num_ = 0;
      for (auto &point : pv_list_) point.normal.setZero();
      position_last_ = state_.pos_end;
      break;
    }
    state_ += solution;
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) { flg_EKF_converged = true; }
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    /*** Rematch Judgement ***/

    if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) { rematch_num++; }

    /*** Convergence Judgements and Covariance Update ***/
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1)))
    {
      /*** Covariance Update ***/
      state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
          (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - G.block<DIM_STATE, DIM_STATE>(0, 0)) * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
      if (state_.visual_reference.valid)
        state_.visual_reference.cross = ((I_STATE - G) * state_.visual_reference.cross).eval();
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
      EKF_stop_flg = true;
    }
    if (EKF_stop_flg) break;
  }

}

void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points)
{
  for (const auto &point : input_points) geometry_map_.insert(point.point_w);
}

void VoxelMapManager::BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list,
                                         bool conservative_matching)
{
  residual_buffer_.resize(pv_list.size());
  residual_valid_.resize(pv_list.size());
  search_caches_.resize(pv_list.size());
  ptpl_list.clear();
  ptpl_list.reserve(pv_list.size());
  const int query_count = static_cast<int>(pv_list.size());

#ifdef MP_EN
#pragma omp parallel for num_threads(std::min(MP_PROC_NUM, omp_get_max_threads())) schedule(static)
#endif
  for (int i = 0; i < query_count; ++i)
  {
    pointWithVar &point = pv_list[i];
    point.point_w = state_.rot_end * point.point_i + state_.pos_end;
    point.normal.setZero();
    residual_valid_[i] = 0;
    const PlaneEstimate plane = geometry_map_.queryPlane(point.point_w, search_caches_[i]);
    if (!plane.withinSupport(point.point_w)) {
      continue;
    }
    const V3D normal_imu = state_.rot_end.transpose() * plane.normal;
    const V3D normal_lidar = extR_.transpose() * normal_imu;
    MD(6, 1) jacobian;
    jacobian.head<3>() = point.point_i.cross(normal_imu);
    jacobian.tail<3>() = plane.normal;
    const double measurement_variance = normal_lidar.dot(point.body_var * normal_lidar);
    const double total_variance = measurement_variance + jacobian.dot(state_.cov.topLeftCorner<6, 6>() * jacobian);
    // A failed local update must not use the broad pose prior to accept distant
    // surfaces again. The prior still participates unchanged in the EKF solve.
    const double gate_variance = config_setting_.variance_floor_ +
        std::max(0.0, conservative_matching ? measurement_variance : total_variance);
    const double distance = plane.signedDistance(point.point_w);
    if (!std::isfinite(total_variance) || !std::isfinite(gate_variance) || gate_variance <= 0.0 ||
        distance * distance > config_setting_.sigma_num_ * config_setting_.sigma_num_ * gate_variance) {
      continue;
    }
    PointToPlane &residual = residual_buffer_[i];
    residual.point_w_ = point.point_w;
    residual.normal_ = plane.normal;
    residual.rotation_jacobian_ = jacobian.head<3>();
    residual.inverse_variance_ = 1.0 / (config_setting_.variance_floor_ + measurement_variance);
    residual.dis_to_plane_ = distance;

    point.normal = plane.normal;
    residual_valid_[i] = 1;
  }
  for (std::size_t i = 0; i < pv_list.size(); ++i)
    if (residual_valid_[i]) ptpl_list.push_back(residual_buffer_[i]);
}

void VoxelMapManager::pubVoxelMap()
{
  if (!config_setting_.is_pub_plane_map_ || voxel_map_pub_.getNumSubscribers() == 0) return;
  visualization_msgs::MarkerArray markers;
  visualization_msgs::Marker clear;
  clear.header.frame_id = "camera_init";
  clear.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(clear);
  int marker_id = 0;
  geometry_map_.visitBlocks([&](const V3D &center)
  {
    const PlaneEstimate plane = geometry_map_.queryPlane(center);
    if (plane.valid) pubSinglePlane(markers, plane, marker_id++);
  });
  voxel_map_pub_.publish(markers);
}

void VoxelMapManager::pubSinglePlane(visualization_msgs::MarkerArray &markers, const PlaneEstimate &estimate, int id)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "camera_init";
  marker.header.stamp = ros::Time();
  marker.ns = "plane";
  marker.id = id;
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = estimate.center.x();
  marker.pose.position.y = estimate.center.y();
  marker.pose.position.z = estimate.center.z();
  M3D rotation;
  rotation.col(0) = estimate.tangent;
  rotation.col(1) = estimate.normal.cross(estimate.tangent);
  rotation.col(2) = estimate.normal;
  const Eigen::Quaterniond quaternion(rotation);
  marker.pose.orientation.x = quaternion.x();
  marker.pose.orientation.y = quaternion.y();
  marker.pose.orientation.z = quaternion.z();
  marker.pose.orientation.w = quaternion.w();
  marker.scale.x = 3.0 * std::sqrt(estimate.eigenvalues[2]);
  marker.scale.y = 3.0 * std::sqrt(estimate.eigenvalues[1]);
  marker.scale.z = std::max(0.001, 2.0 * std::sqrt(estimate.eigenvalues[0]));
  uint8_t r, g, b;
  mapJet(0.0, 0.0, 1.0, r, g, b);
  marker.color.a = 0.8;
  marker.color.r = r / 255.0;
  marker.color.g = g / 255.0;
  marker.color.b = b / 255.0;
  markers.markers.push_back(marker);
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}
