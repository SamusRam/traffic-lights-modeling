from filterpy.kalman import KalmanFilter
from scipy.stats import multivariate_normal
import numpy as np
import os


class KalmanTrackerPredictor(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, init_measurements,
                 measured_noise_x_coordinate_cov_adjustment,
                 measured_noise_y_coordinate_cov_adjustment,
                 measured_noise_x_speed_coordinate_cov_adjustment,
                 measured_noise_y_speed_coordinate_cov_adjustment,
                 x_coordinate_speed_cov_adjustment,
                 y_coordinate_speed_cov_adjustment,
                 x_coordinate_cov_adjustment,
                 y_coordinate_cov_adjustment,
                 x_coordinate_acceleration_adjustment,
                 y_coordinate_acceleration_adjustment,
                 noise_in_x_coordinate_speed_cov_adjustment,
                 noise_in_y_coordinate_speed_cov_adjustment,
                 noise_in_x_coordinate_cov_adjustment,
                 noise_in_y_coordinate_cov_adjustment,
                 noise_in_x_coordinate_acceleration_adjustment,
                 noise_in_y_coordinate_acceleration_adjustment,
                 propagate_noises=True,
                 use_coordinates_for_likelihood=True,
                 fps=None, track_id=None, timestamp=None):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant acceleration model
        self.kf = KalmanFilter(dim_x=6,
                               dim_z=4)
        dt = 1 / fps if not fps is None else 1
        self.z_cov = self.z_mean = None
        self.appearance_emb = None
        self.kf.F = np.array(
            [[1, 0, dt, 0, .5 * dt ** 2, 0],  # coord1
             [0, 1, 0, dt, 0, .5 * dt ** 2],  # coord2
             [0, 0, 1, 0, dt, 0],  # coord1 speed
             [0, 0, 0, 1, 0, dt],  # coord2 speed
             [0, 0, 0, 0, 1, 0],  # coord1 acceleration
             [0, 0, 0, 0, 0, 1],  # coord2 acceleration
             ])  # self-likeness
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]])

        def adjust_state_covariance_matrix(cov, x_coordinate_speed_adjustment, y_coordinate_speed_adjustment,
                                           x_coordinate_acceleration_adjustment,
                                           y_coordinate_acceleration_adjustment,
                                           x_coordinate_adjustment, y_coordinate_adjustment):
            cov[0, 0] *= x_coordinate_adjustment
            cov[1, 1] *= y_coordinate_adjustment
            cov[2, 2] *= x_coordinate_speed_adjustment
            cov[3, 3] *= y_coordinate_speed_adjustment
            cov[4, 4] *= x_coordinate_acceleration_adjustment
            cov[5, 5] *= y_coordinate_acceleration_adjustment
            return cov

        def adjust_measurement_covariance_matrix(cov,
                                                 x_coordinate_adjustment, y_coordinate_adjustment,
                                                 x_speed_coordinate_adjustment, y_speed_coordinate_adjustment):
            cov[0, 0] *= x_coordinate_adjustment
            cov[1, 1] *= y_coordinate_adjustment
            cov[2, 2] *= x_speed_coordinate_adjustment
            cov[3, 3] *= y_speed_coordinate_adjustment
            return cov

        self.kf.R = adjust_measurement_covariance_matrix(self.kf.R,
                                                         measured_noise_x_coordinate_cov_adjustment,
                                                         measured_noise_y_coordinate_cov_adjustment,
                                                         measured_noise_x_speed_coordinate_cov_adjustment,
                                                         measured_noise_y_speed_coordinate_cov_adjustment)

        self.kf.P = adjust_state_covariance_matrix(self.kf.P, x_coordinate_speed_cov_adjustment,
                                                   y_coordinate_speed_cov_adjustment,
                                                   x_coordinate_acceleration_adjustment,
                                                   y_coordinate_acceleration_adjustment,
                                                   x_coordinate_cov_adjustment,
                                                   y_coordinate_cov_adjustment)

        self.kf.Q = adjust_state_covariance_matrix(self.kf.Q, noise_in_x_coordinate_speed_cov_adjustment,
                                                   noise_in_y_coordinate_speed_cov_adjustment,
                                                   noise_in_x_coordinate_acceleration_adjustment,
                                                   noise_in_y_coordinate_acceleration_adjustment,
                                                   noise_in_x_coordinate_cov_adjustment,
                                                   noise_in_y_coordinate_cov_adjustment)

        self.kf.x[:-2] = init_measurements.reshape(-1, 1)
        self.time_since_update = 0
        self.track_id = track_id
        self.timestamp = timestamp
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.propagate_noises = propagate_noises
        self.use_coordinates_for_likelihood = use_coordinates_for_likelihood
        self.gt_id_of_last_measurement = None
        self.derive_measurement_estimation()

    def update(self, state):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(state)

    def predict(self):
        self.kf.predict()
        self.derive_measurement_estimation()
        self.age += 1

    def process_history(self, history_measurements):
        for hist_idx in range(history_measurements.shape[0] - 1, 0, -1):
            self.predict()
            self.update(history_measurements[hist_idx])

    def predict_future_positions(self, n_steps=50):
        future_coordinates = []
        for step in range(n_steps):
            self.predict()
            future_coordinates.append(self.z_mean[:2])
        return np.expand_dims(np.vstack(future_coordinates), 0)

    def get_state(self):
        """
        Returns the current estimate.
        """
        return self.kf.x

    def get_prediction(self):
        return self.z_mean

    def derive_measurement_estimation(self):
        # must be called after current iteration predict
        if self.propagate_noises:
            self.z_cov = np.matmul(np.matmul(self.kf.H, self.kf.P + self.kf.Q), self.kf.H.T)[:4, :4] + self.kf.R[:4, :4]
            self.z_mean = np.matmul(self.kf.H, self.kf.x)[:4].ravel()
        else:
            self.z_cov = np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T)[:4, :4]
            self.z_mean = np.matmul(self.kf.H, self.kf.x)[:4].ravel()

    def get_position_estimation_with_std(self):
        cv2_x_mean = self.z_mean[1]
        cv2_y_mean = self.z_mean[0]
        cv2_x_std = np.sqrt(self.z_cov[1, 1])
        cv2_y_std = np.sqrt(self.z_cov[0, 0])
        # print(cv2_y_std, cv2_x_std)
        # # manually setting correspondence with the original sort buggy approach
        # cv2_x_mean = self.kf.x[1][0]
        # cv2_y_mean = self.kf.x[0][0]
        # cv2_x_std = np.sqrt(self.kf.P[1, 1])
        # cv2_y_std = np.sqrt(self.kf.P[0, 0])
        return [(cv2_x_mean, cv2_y_mean), (cv2_x_std, cv2_y_std)]
