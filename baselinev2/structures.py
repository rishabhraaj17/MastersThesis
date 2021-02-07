from typing import Union, List


class ObjectFeatures(object):
    def __init__(self, idx, xy, past_xy, final_xy, flow, past_flow, past_bbox, final_bbox, frame_number, history=None,
                 is_track_live=True, gt_history=None, track_direction=None, velocity_history=None,
                 velocity_direction=None, running_velocity=None, per_step_distance=None):
        super(ObjectFeatures, self).__init__()
        self.idx = idx
        self.xy = xy
        self.flow = flow
        self.past_bbox = past_bbox
        self.final_bbox = final_bbox
        self.past_flow = past_flow
        self.final_xy = final_xy
        self.past_xy = past_xy
        self.is_track_live = is_track_live
        self.frame_number = frame_number
        self.gt_box = None
        self.past_gt_box = None
        self.gt_track_idx = None
        self.gt_past_current_distance = None
        self.gt_history = gt_history
        self.track_history = history
        self.track_direction = track_direction
        self.velocity_history = velocity_history
        self.velocity_direction = velocity_direction
        self.per_step_distance = per_step_distance
        self.running_velocity = running_velocity

    def __eq__(self, other):
        return self.idx == other.idx


class MinimalObjectFeatures(object):
    def __init__(self, idx, past_bbox, final_bbox, frame_number, flow, past_flow, is_track_live=True, gt_history=None,
                 track_direction=None, velocity_history=None, velocity_direction=None, history=None):
        super(MinimalObjectFeatures, self).__init__()
        self.idx = idx
        self.past_bbox = past_bbox
        self.final_bbox = final_bbox
        self.is_track_live = is_track_live
        self.frame_number = frame_number
        self.gt_box = None
        self.past_gt_box = None
        self.gt_track_idx = None
        self.gt_history = gt_history
        self.track_history = history
        self.track_direction = track_direction
        self.velocity_history = velocity_history
        self.velocity_direction = velocity_direction
        self.flow = flow
        self.past_flow = past_flow

    def __eq__(self, other):
        return self.idx == other.idx


class AgentFeatures(object):
    def __init__(self, track_idx, activations_t, activations_t_minus_one, activations_t_plus_one, future_flow,
                 past_flow, bbox_t, bbox_t_plus_one, bbox_t_minus_one, frame_number, activations_future_frame,
                 activations_past_frame, final_features_future_activations, is_track_live=True, gt_box=None,
                 past_gt_box=None, gt_track_idx=None, gt_past_current_distance=None, frame_number_t=None,
                 frame_number_t_minus_one=None, frame_number_t_plus_one=None, past_frames_used_in_of_estimation=None,
                 frame_by_frame_estimation=False, future_frames_used_in_of_estimation=None, future_gt_box=None,
                 past_gt_track_idx=None, future_gt_track_idx=None, gt_current_future_distance=None,
                 past_box_inconsistent=False, future_box_inconsistent=False, gt_history=None, history=None,
                 track_direction=None, velocity_history=None, velocity_direction=None):
        super(AgentFeatures, self).__init__()
        self.frame_number_t = frame_number_t
        self.frame_number_t_minus_one = frame_number_t_minus_one
        self.frame_number_t_plus_one = frame_number_t_plus_one
        self.past_frames_used_in_of_estimation = past_frames_used_in_of_estimation
        self.future_frames_used_in_of_estimation = future_frames_used_in_of_estimation
        self.frame_by_frame_estimation = frame_by_frame_estimation
        self.track_idx = track_idx
        self.activations_t = activations_t
        self.future_flow = future_flow
        self.bbox_t = bbox_t
        self.bbox_t_plus_one = bbox_t_plus_one
        self.bbox_t_minus_one = bbox_t_minus_one
        self.past_flow = past_flow
        self.activations_t_plus_one = activations_t_plus_one
        self.activations_t_minus_one = activations_t_minus_one
        self.is_track_live = is_track_live
        self.frame_number = frame_number
        self.activations_past_frame = activations_past_frame
        self.activations_future_frame = activations_future_frame
        self.final_features_future_activations = final_features_future_activations
        self.gt_box = gt_box
        self.future_gt_box = future_gt_box
        self.past_gt_box = past_gt_box
        self.gt_track_idx = gt_track_idx
        self.future_gt_track_idx = future_gt_track_idx
        self.past_gt_track_idx = past_gt_track_idx
        self.gt_past_current_distance = gt_past_current_distance
        self.gt_current_future_distance = gt_current_future_distance
        self.future_box_inconsistent = future_box_inconsistent
        self.past_box_inconsistent = past_box_inconsistent
        self.gt_history = gt_history
        self.track_history = history
        self.track_direction = track_direction
        self.velocity_history = velocity_history
        self.velocity_direction = velocity_direction

    def __eq__(self, other):
        return self.track_idx == other.track_idx


class FrameFeatures(object):
    def __init__(self, frame_number: int, object_features: Union[List[ObjectFeatures], List[AgentFeatures],
                                                                 List[MinimalObjectFeatures]]
                 , flow=None, past_flow=None):
        super(FrameFeatures, self).__init__()
        self.frame_number = frame_number
        self.object_features = object_features
        self.flow = flow
        self.past_flow = past_flow


class TrackFeatures(object):
    def __init__(self, track_id: int):
        super(TrackFeatures, self).__init__()
        self.track_id = track_id
        self.object_features: Union[List[ObjectFeatures], List[AgentFeatures]] = []

    def __eq__(self, other):
        return self.track_id == other.track_id


class Track(object):
    def __init__(self, bbox, idx, history=None, gt_track_idx=None):
        super(Track, self).__init__()
        self.idx = idx
        self.bbox = bbox
        self.gt_track_idx = gt_track_idx
        self.history = history
