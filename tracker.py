import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Bounding boxes are in [x1, y1, x2, y2] format.
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)
    
    # Calculate intersection area
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    # Calculate union area
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def bbox_to_z(bbox):
    """
    Converts bounding box in [x1, y1, x2, y2] format to measurement vector z
    in [u, v, s, r]^T format, where:
      u, v = center coordinates
      s = area (scale)
      r = aspect ratio (width / height)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 0.0
    return np.array([[u], [v], [s], [r]])

def x_to_bbox(x):
    """
    Converts state vector x [u, v, s, r, u_dot, v_dot, s_dot]^T to [x1, y1, x2, y2] bounding box.
    """
    u, v, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
    s = max(0.0, s)
    r = max(1e-6, r)
    w = np.sqrt(s * r)
    h = np.sqrt(s / r)
    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0
    return np.array([x1, y1, x2, y2])

class KalmanBoxTracker:
    """
    Represents the state of a single tracked object using a Kalman Filter.
    """
    count = 0
    
    def __init__(self, bbox, cls_id=0, conf=0.0):
        # State vector x: [u, v, s, r, u_dot, v_dot, s_dot]^T
        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)
        
        # State covariance matrix P (uncertainty)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # Assign high uncertainty to velocities initially
        
        # State transition matrix F (constant velocity model)
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
            
        # Measurement matrix H
        self.H = np.zeros((4, 7))
        for i in range(4):
            self.H[i, i] = 1.0
            
        # Measurement noise covariance matrix R
        self.R = np.eye(4)
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0
        
        # Process noise covariance matrix Q
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01
        self.Q[3, 3] *= 0.01
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.history = []
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        
        self.cls_id = cls_id
        self.conf = conf

    def predict(self):
        """
        Advances the state vector using the constant velocity motion model.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        
        self.history.append(x_to_bbox(self.x))
        return self.history[-1]

    def update(self, bbox, cls_id=None, conf=None):
        """
        Updates the state vector with the observed bounding box.
        """
        self.time_since_update = 0
        self.hits += 1
        self.history = []
        
        if cls_id is not None:
            self.cls_id = cls_id
        if conf is not None:
            self.conf = conf
            
        # Kalman Update step
        z = bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(7) - np.dot(K, self.H), self.P)
        
        return x_to_bbox(self.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return x_to_bbox(self.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Associates detections to predicted trackers using the Hungarian Algorithm.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det[:4], trk[:4])
            
    # Solve matching problem using linear_sum_assignment (Hungarian Algorithm)
    # Minimizing (1.0 - IoU) matches detections and predictions with the highest IoU
    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)
    
    matched_indices = np.stack((row_ind, col_ind), axis=1)
    
    # Identify unmatched detections and trackers
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
            
    # Filter matches that do not meet the minimum IoU threshold
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort:
    """
    Manages multiple KalmanBoxTrackers and associates new detections across frames.
    """
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Updates the tracker with detections from the current frame.
        dets - a numpy array of detections in the format [[x1, y1, x2, y2, confidence, class_id], ...]
        Returns a similar array with added tracker IDs: [[x1, y1, x2, y2, tracker_id, class_id, confidence], ...]
        """
        self.frame_count += 1
        
        # 1. Predict next state for existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove trackers that returned invalid predictions (NaNs)
        for index in sorted(to_del, reverse=True):
            self.trackers.pop(index)
            
        # Re-build trackers list representation
        trks = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(self.trackers):
            pos = trk.get_state()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            
        # 2. Associate detections with predicted states
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        
        # 3. Update matched trackers with actual detections
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(
                dets[det_idx, :4], 
                cls_id=int(dets[det_idx, 5]), 
                conf=dets[det_idx, 4]
            )
            
        # 4. Initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :4], 
                cls_id=int(dets[i, 5]), 
                conf=dets[i, 4]
            )
            self.trackers.append(trk)
            
        # 5. Retrieve active tracks and remove dead trackers
        ret = []
        active_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                active_trackers.append(trk)
                # Ensure the track has been active long enough to be reported
                if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                    d = trk.get_state()
                    ret.append(np.concatenate((d, [trk.id, trk.cls_id, trk.conf])))
                    
        self.trackers = active_trackers
        
        if len(ret) > 0:
            return np.stack(ret)
        return np.empty((0, 7))
