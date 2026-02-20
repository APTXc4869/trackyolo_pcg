# Ship-ByteTrack (compact reference) with IMM Kalman filter and Hungarian association.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b
    w = x2 - x1
    h = y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w, h], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return 1.0 - float(np.dot(a, b) / (na * nb))


@dataclass
class Detection:
    xyxy: np.ndarray
    score: float
    cls: int
    feat: Optional[np.ndarray] = None


class KalmanCV:
    def __init__(self):
        self.ndim = 4
        self.dt = 1.0
        self._motion_mat = np.eye(2*self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.eye(self.ndim, 2*self.ndim, dtype=np.float32)

    def initiate(self, z):
        mean = np.zeros((2*self.ndim,), dtype=np.float32)
        mean[:self.ndim] = z
        cov = np.eye(2*self.ndim, dtype=np.float32)
        cov[:self.ndim, :self.ndim] *= 1.0
        cov[self.ndim:, self.ndim:] *= 0.5
        return mean, cov

    def predict(self, mean, cov, q_scale=1e-2):
        q = np.eye(2*self.ndim, dtype=np.float32) * q_scale
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + q
        return mean, cov

    def project(self, mean, cov, r_scale=1e-1):
        r = np.eye(self.ndim, dtype=np.float32) * r_scale
        pm = self._update_mat @ mean
        pc = self._update_mat @ cov @ self._update_mat.T + r
        return pm, pc

    def update(self, mean, cov, z):
        pm, pc = self.project(mean, cov)
        k = cov @ self._update_mat.T @ np.linalg.inv(pc)
        innovation = z - pm
        new_mean = mean + k @ innovation
        new_cov = cov - k @ pc @ k.T
        return new_mean, new_cov

    def likelihood(self, mean, cov, z):
        pm, pc = self.project(mean, cov)
        diff = z - pm
        inv = np.linalg.pinv(pc)
        exponent = -0.5 * float(diff.T @ inv @ diff)
        det = max(np.linalg.det(pc), 1e-12)
        norm = 1.0 / math.sqrt(((2*math.pi) ** self.ndim) * det)
        return norm * math.exp(exponent)


class IMMFilter:
    def __init__(self, p11=0.9, p22=0.9):
        self.kf_cv = KalmanCV()
        self.kf_ca = KalmanCV()
        self.P = np.array([[p11, 1-p11], [1-p22, p22]], dtype=np.float32)
        self.mu0 = np.array([0.5, 0.5], dtype=np.float32)

    def initiate(self, z):
        m1, c1 = self.kf_cv.initiate(z)
        m2, c2 = self.kf_ca.initiate(z)
        return [m1, m2], [c1, c2], self.mu0.copy()

    def mix(self, means, covs, mu):
        c = self.P.T @ mu
        c = np.clip(c, 1e-12, None)
        mu_ij = (self.P * mu.reshape(1, -1)) / c.reshape(-1, 1)

        mixed_means, mixed_covs = [], []
        for j in range(2):
            m = mu_ij[j, 0] * means[0] + mu_ij[j, 1] * means[1]
            dm0 = (means[0] - m).reshape(-1, 1)
            dm1 = (means[1] - m).reshape(-1, 1)
            C = mu_ij[j, 0] * (covs[0] + dm0 @ dm0.T) + mu_ij[j, 1] * (covs[1] + dm1 @ dm1.T)
            mixed_means.append(m)
            mixed_covs.append(C)
        return mixed_means, mixed_covs, c / np.sum(c)

    def predict(self, means, covs, mu):
        means, covs, mu = self.mix(means, covs, mu)
        means[0], covs[0] = self.kf_cv.predict(means[0], covs[0], q_scale=1e-2)
        means[1], covs[1] = self.kf_ca.predict(means[1], covs[1], q_scale=5e-2)
        return means, covs, mu

    def update(self, means, covs, mu, z):
        l1 = self.kf_cv.likelihood(means[0], covs[0], z)
        l2 = self.kf_ca.likelihood(means[1], covs[1], z)
        lik = np.array([l1, l2], dtype=np.float32)
        mu = mu * lik
        mu = mu / (np.sum(mu) + 1e-12)

        means[0], covs[0] = self.kf_cv.update(means[0], covs[0], z)
        means[1], covs[1] = self.kf_ca.update(means[1], covs[1], z)

        mean = mu[0] * means[0] + mu[1] * means[1]
        dm0 = (means[0] - mean).reshape(-1, 1)
        dm1 = (means[1] - mean).reshape(-1, 1)
        cov = mu[0] * (covs[0] + dm0 @ dm0.T) + mu[1] * (covs[1] + dm1 @ dm1.T)
        return means, covs, mu, mean, cov


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    means: List[np.ndarray]
    covs: List[np.ndarray]
    mu: np.ndarray
    cls: int
    score: float
    feat: Optional[np.ndarray]
    age: int = 0
    time_since_update: int = 0
    hits: int = 1
    alive: bool = True

    def to_xyxy(self):
        x, y, w, h = self.mean[:4]
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2], dtype=np.float32)


class ShipByteTrack:
    def __init__(self, track_high=0.6, track_low=0.1, new_track=0.7, track_buffer=30, match_thresh=0.7,
                 use_reid=False, reid_weight=0.5):
        self.track_high = track_high
        self.track_low = track_low
        self.new_track = new_track
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.use_reid = use_reid
        self.reid_weight = reid_weight

        self.imm = IMMFilter()
        self.tracks: List[Track] = []
        self.next_id = 1

    def _associate(self, tracks: List[Track], dets: List[Detection]):
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        cost = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            tb = t.to_xyxy()
            for j, d in enumerate(dets):
                c = 1.0 - iou_xyxy(tb, d.xyxy)
                if self.use_reid and t.feat is not None and d.feat is not None:
                    c_reid = cosine_distance(t.feat, d.feat)
                    c = (1.0 - self.reid_weight) * c + self.reid_weight * c_reid
                cost[i, j] = c

        r, c = linear_sum_assignment(cost)
        matches = []
        ut = set(range(len(tracks)))
        ud = set(range(len(dets)))
        for i, j in zip(r, c):
            if cost[i, j] > (1.0 - self.match_thresh):
                continue
            matches.append((i, j))
            ut.discard(i)
            ud.discard(j)
        return matches, sorted(ut), sorted(ud)

    def update(self, detections: List[Detection]) -> List[Track]:
        high = [d for d in detections if d.score >= self.track_high]
        low = [d for d in detections if self.track_low <= d.score < self.track_high]

        for t in self.tracks:
            t.means, t.covs, t.mu = self.imm.predict(t.means, t.covs, t.mu)
            t.mean = t.mu[0] * t.means[0] + t.mu[1] * t.means[1]
            t.age += 1
            t.time_since_update += 1

        matches, ut, ud = self._associate(self.tracks, high)
        for ti, di in matches:
            tr = self.tracks[ti]
            det = high[di]
            z = xyxy_to_xywh(det.xyxy)
            tr.means, tr.covs, tr.mu, tr.mean, tr.cov = self.imm.update(tr.means, tr.covs, tr.mu, z)
            tr.score = det.score
            tr.cls = det.cls
            tr.feat = det.feat if det.feat is not None else tr.feat
            tr.hits += 1
            tr.time_since_update = 0

        rem_tracks = [self.tracks[i] for i in ut]
        matches2, ut2, ud2 = self._associate(rem_tracks, low)
        for ti, di in matches2:
            tr = rem_tracks[ti]
            det = low[di]
            z = xyxy_to_xywh(det.xyxy)
            tr.means, tr.covs, tr.mu, tr.mean, tr.cov = self.imm.update(tr.means, tr.covs, tr.mu, z)
            tr.score = det.score
            tr.cls = det.cls
            tr.feat = det.feat if det.feat is not None else tr.feat
            tr.hits += 1
            tr.time_since_update = 0

        for t in self.tracks:
            if t.time_since_update > self.track_buffer:
                t.alive = False
        self.tracks = [t for t in self.tracks if t.alive]

        for det in [high[i] for i in ud]:
            if det.score < self.new_track:
                continue
            z = xyxy_to_xywh(det.xyxy)
            means, covs, mu = self.imm.initiate(z)
            mean = mu[0] * means[0] + mu[1] * means[1]
            cov = mu[0] * covs[0] + mu[1] * covs[1]
            self.tracks.append(Track(self.next_id, mean, cov, means, covs, mu, det.cls, det.score, det.feat))
            self.next_id += 1

        return list(self.tracks)
