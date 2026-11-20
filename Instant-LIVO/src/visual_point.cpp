/* 
This file is part of Instant-LIVO

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "visual_point.h"
#include "vio.h"
#include <algorithm>
#include "feature.h"
#include <stdexcept>
#include <vikit/math_utils.h>

VisualPoint::VisualPoint(const Vector3d &pos)
    : pos_(pos), previous_normal_(Vector3d::Zero()), normal_(Vector3d::Zero()),
      is_converged_(false), is_normal_initialized_(false), has_ref_patch_(false)
{
}

VisualPoint::~VisualPoint() 
{
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    delete(*it);
  }
  obs_.clear();
  ref_patch = nullptr;
}

void VisualPoint::addFrameRef(Feature *ftr)
{
  obs_.push_front(ftr);
}

void VisualPoint::deleteFeatureRef(Feature *ftr)
{
  if (ref_patch == ftr)
  {
    ref_patch = nullptr;
    has_ref_patch_ = false;
  }
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    if ((*it) == ftr)
    {
      delete((*it));
      obs_.erase(it);
      return;
    }
  }
}

bool VisualPoint::getCloseViewObs(const Vector3d &framepos, Feature *&ftr, const Vector2d &cur_px) const
{
  // TODO: get frame with same point of view AND same pyramid level!
  if (obs_.size() <= 0) return false;

  Vector3d obs_dir(framepos - pos_);
  obs_dir.normalize();
  auto min_it = obs_.begin();
  double min_cos_angle = 0;
  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    Vector3d dir((*it)->T_f_w_.inverse().translation() - pos_);
    dir.normalize();
    double cos_angle = obs_dir.dot(dir);
    if (cos_angle > min_cos_angle)
    {
      min_cos_angle = cos_angle;
      min_it = it;
    }
  }
  ftr = *min_it;

  // Vector2d ftr_px = ftr->px_;
  // double pixel_dist = (cur_px-ftr_px).norm();

  // if(pixel_dist > 200)
  // {
  //   ROS_ERROR("The pixel dist exceeds 200.");
  //   return false;
  // }

  if (min_cos_angle < 0.5) // assume that observations larger than 60° are useless 0.5
  {
    // ROS_ERROR("The obseved angle is larger than 60°.");
    return false;
  }

  return true;
}

void VisualPoint::findMinScoreFeature(const Vector3d &framepos, Feature *&ftr) const
{
  auto min_it = obs_.begin();
  float min_score = std::numeric_limits<float>::max();

  for (auto it = obs_.begin(), ite = obs_.end(); it != ite; ++it)
  {
    if ((*it)->score_ < min_score)
    {
      min_score = (*it)->score_;
      min_it = it;
    }
  }
  ftr = *min_it;
}

void VisualPoint::deleteNonRefPatchFeatures()
{
  for (auto it = obs_.begin(); it != obs_.end();)
  {
    if (*it != ref_patch)
    {
      delete *it;
      it = obs_.erase(it);
    }
    else
    {
      ++it;
    }
  }
}

void VIOManager::generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg)
{
  reference_track_count = std::max(reference_track_count, total_points);
  if (!needsReferenceRefresh() || pg.empty()) return;

  // Build candidates separately; the active reference remains available for plotting.
  const int tracked_points = total_points;
  resetGrid();
  total_points = tracked_points;
  feat_map.swap(reference_candidates);
  const M3D lidar_to_world = state->rot_end * Rli.transpose();
  const V3D origin = state->pos_end - lidar_to_world * Pli;

  // double t0 = omp_get_wtime();
  for (int i = 0; i < pg.size(); i++)
  {
    if (!pg[i].normal.allFinite() || pg[i].normal.squaredNorm() < 1e-12) continue;

    V3D pt = lidar_to_world * pg[i].point_b + origin;
    V3D pt_cam(new_frame_->w2f(pt));
    if (!pt_cam.allFinite() || pt_cam.z() < 1.0 || pt_cam.z() > 80.0) continue;
    V2D pc(cam->world2cam(pt_cam));

    if (pc.allFinite() && new_frame_->cam_->isInFrame(pc.cast<int>(), border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        if (cur_value <= 1.0f || pt_cam.z() > map_dist[index] + 0.1) continue;
        if (pt_cam.z() < map_dist[index] - 0.1 || cur_value > scan_value[index])
        {
          map_dist[index] = pt_cam.z();
          scan_value[index] = cur_value;
          append_voxel_points[index] = pg[i];
          append_voxel_points[index].point_w = pt;
          grid_num[index] = TYPE_POINTCLOUD;
        }
      }
    }
  }

  // double t_b1 = omp_get_wtime() - t0;
  // t0 = omp_get_wtime();

  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (add >= max_visual_points) break;
    if (grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = append_voxel_points[i];
      // Reference-pose noise is handled separately by the filter.
      pt_var.var = lidar_to_world * pt_var.body_var * lidar_to_world.transpose();
      V3D pt = pt_var.point_w;

      V3D norm_vec(new_frame_->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(new_frame_->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(new_frame_->w2c(pt));

      float *patch = new float[patch_size_total];
      getImagePatch(img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_, 0);
      ftr_new->img_ = img;
      ftr_new->id_ = new_frame_->id_;
      ftr_new->inv_expo_time_ = state->inv_expo_time;

      pt_new->addFrameRef(ftr_new);
      pt_new->ref_patch = ftr_new;
      pt_new->has_ref_patch_ = true;
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }

      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      add += 1;
      // map_cur_frame.push_back(pt_new);
    }
  }

  feat_map.swap(reference_candidates);

  // double t_b2 = omp_get_wtime() - t0;

  printf("[ VIO ] Append %d new visual map points\n", add);
  // printf("pg.size: %d \n", pg.size());
  // printf("B1. : %.6lf \n", t_b1);
  // printf("B2. : %.6lf \n", t_b2);
}

bool VIOManager::needsReferenceRefresh() const
{
  constexpr double min_overlap = 0.6;
  constexpr double max_displacement = 12.0;
  if (reference_frame_id < 0 || new_frame_->id_ - reference_frame_id >= reference_max_age ||
      total_points < std::max(min_reference_points, int(min_overlap * reference_track_count)))
    return true;

  std::vector<double> displacements;
  displacements.reserve(total_points);
  for (const auto *point : visual_submap->voxel_points)
  {
    const V2D pixel = new_frame_->w2c(point->pos_);
    if (pixel.allFinite())
      displacements.push_back((pixel - point->ref_patch->px_).squaredNorm());
  }
  if (displacements.empty()) return true;
  auto median = displacements.begin() + displacements.size() / 2;
  std::nth_element(displacements.begin(), median, displacements.end());
  return *median >= max_displacement * max_displacement;
}

void VIOManager::updateVisualMapPoints(cv::Mat img)
{
  if (total_points == 0 && reference_candidates.empty()) return;

  int update_num = 0;
  for (const auto &entry : reference_candidates) update_num += entry.second->voxel_points.size();
  if (update_num >= min_reference_points)
  {
    // Release borrowed points before replacing the active reference.
    visual_submap->reset();
    sub_feat_map.clear();
    resetGrid();
    has_ref_patch_cache = false;
    for (auto& pair : warp_map) delete pair.second;
    warp_map.clear();
    feat_map.swap(reference_candidates);
    reference_frame_id = new_frame_->id_;
    reference_track_count = 0;
  }
  else update_num = 0;

  // On rejection these are candidates; on success these are the old reference.
  for (auto& pair : reference_candidates) delete pair.second;
  reference_candidates.clear();
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const OctVoxMap &plane_map)
{
  if (reference_frame_id != new_frame_->id_) return;

  // Exposure is relative to the new reference; transform its covariance too.
  const double gain = state->inv_expo_time;
  if (exposure_estimate_en)
  {
    for (auto &entry : feat_map)
      for (auto *point : entry.second->voxel_points)
        point->ref_patch->inv_expo_time_ /= gain;
    state->inv_expo_time = 1.0;
    state->cov.row(6) /= gain;
    state->cov.col(6) /= gain;
  }
  state->visual_reference.capture(state->cov, state->rot_end, state->pos_end);
}
