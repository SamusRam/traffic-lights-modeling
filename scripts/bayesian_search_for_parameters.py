import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import os
    import sys

    if os.name == 'nt':
        from scripts.l5kit_modified.l5kit_modified import AgentDatasetModified, create_chopped_dataset
        from scripts.kalman import KalmanTrackerPredictor

        os.environ["L5KIT_DATA_FOLDER"] = "input/"
    elif os.name == 'posix':
        from l5kit_modified import AgentDatasetModified, create_chopped_dataset
        from kalman import KalmanTrackerPredictor

        os.environ["L5KIT_DATA_FOLDER"] = "../input/"
    else:
        sys.exit('Unsupported platform')
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    import pickle
    import logging
    from logging.handlers import RotatingFileHandler
    import sys
    import time
    import numpy as np
    import argparse
    import json
    from tqdm.auto import tqdm
    from l5kit.evaluation import write_pred_csv, compute_metrics_csv
    from l5kit.evaluation.metrics import neg_multi_log_likelihood
    from l5kit.data import LocalDataManager, ChunkedDataset
    from torch.utils.data import DataLoader
    from multiprocessing import Pool, cpu_count
    from functools import partial
    from pathlib import Path

FILTER_AGENTS_THRESHOLD = 0.5
NUM_FRAMES_TO_CHOP = 100
FUTURE_NUM_FRAMES = 50
MIN_FUTURE_STEPS = 10
BATCH_SIZE = 256
NUM_WORKERS = cpu_count() - 1


def compute_track_predictions(batch_i, data, params):
    history_measurements = data[batch_i]['history_positions']
    params.update({'fps': 10})
    kalman_tracker_predictor = KalmanTrackerPredictor(**params)
    kalman_tracker_predictor.process_history(history_measurements, data[batch_i]['history_availabilities'])
    future_coords_offsets = kalman_tracker_predictor.predict_future_positions()
    timestamp = data[batch_i]['timestamp']
    agent_id = data[batch_i]['track_id']
    return future_coords_offsets, timestamp, agent_id


class KalmanInitializationOptimizer:

    def __init__(self, output_path_root, logging_path,
                 dataset_path,
                 n_optimizer_calls=150,
                 experiment_name='',
                 cov_adjustment_range=(0.001, 10000),
                 debug=False,
                 batch_size=None,
                 num_workers=None,
                 percentage_of_dataset_scenes=1.0):
        self.counter = 0
        self.dimensions = [Integer(50, NUM_FRAMES_TO_CHOP, name='history_num_frames'),
                           Integer(1, 5, name='history_step_size')]
        cov_adjustment_parameters = ['measured_noise_x_coordinate_cov_adjustment',
                                     'measured_noise_y_coordinate_cov_adjustment',
                                     'x_coordinate_speed_cov_adjustment',
                                     'y_coordinate_speed_cov_adjustment',
                                     'x_coordinate_cov_adjustment',
                                     'y_coordinate_cov_adjustment',
                                     'noise_in_x_coordinate_speed_cov_adjustment',
                                     'noise_in_y_coordinate_speed_cov_adjustment',
                                     'noise_in_x_coordinate_cov_adjustment',
                                     'noise_in_y_coordinate_cov_adjustment']

        for cov_adjustment_param in cov_adjustment_parameters:
            self.dimensions.append(Real(*cov_adjustment_range, 'log-uniform', name=cov_adjustment_param))
        time_str = time.strftime('%Y%m%d-%H%M%S')
        self.experiment_name = f'{experiment_name}_{time_str}'
        self.dataset_path = dataset_path
        self.dm = LocalDataManager(None)

        self.output_path = os.path.join(output_path_root, f'experiment_{self.experiment_name}')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.n_optimizer_calls = n_optimizer_calls if not debug else 10
        self.batch_size = batch_size if batch_size is not None else BATCH_SIZE
        self.num_workers = num_workers if num_workers is not None else NUM_WORKERS
        # Setting up logging
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
        self.logger = logging.getLogger('Optimization of Kalman init')
        self.logger.setLevel(logging.INFO)
        fh = RotatingFileHandler(os.path.join(logging_path, f'init_parameters_optimization_{time_str}.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.logger.info(
            f"Using the following dataset: {self.dataset_path}")
        self.debug = debug
        self.eval_base_path = create_chopped_dataset(self.dm.require(self.dataset_path),
                                                     FILTER_AGENTS_THRESHOLD,
                                                     NUM_FRAMES_TO_CHOP,
                                                     FUTURE_NUM_FRAMES,
                                                     MIN_FUTURE_STEPS,
                                                     percentage_of_dataset_scenes)
        self.logger.info(
            f"eval_base_path: {self.eval_base_path}")
        self.eval_gt_path = os.path.join(self.eval_base_path, 'gt.csv')

        eval_zarr_path = str(Path(self.eval_base_path) / Path(self.dm.require(self.dataset_path)).name)
        eval_mask_path = str(Path(self.eval_base_path) / "mask.npz")
        self.eval_zarr = ChunkedDataset(eval_zarr_path).open()
        self.eval_mask = np.load(eval_mask_path)["arr_0"]

    # The objective function to be maximized: avg track duration
    def make_objective(self):
        # This decorator converts your objective function with named arguments into one that
        # accepts a list as argument, while doing the conversion automatically.
        @use_named_args(self.dimensions)
        def objective(**params):
            cfg = {"model_params": {"history_num_frames": params["history_num_frames"],
                                    "history_step_size": params["history_step_size"],
                                    "future_num_frames": FUTURE_NUM_FRAMES,
                                    "future_step_size": 1},
                   "raster_params": {"filter_agents_threshold": FILTER_AGENTS_THRESHOLD}}
            del params["history_num_frames"]
            del params["history_step_size"]


            # ===== INIT DATASET AND LOAD MASK
            dataset = AgentDatasetModified(cfg, self.eval_zarr, agents_mask=self.eval_mask)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    collate_fn=lambda x: x)
            future_coords_offsets = []
            timestamps = []
            agent_ids = []
            for data in tqdm(dataloader, desc='Iterating over data...'):
                compute_track_predictions_partial = partial(compute_track_predictions,
                                                            data=data,
                                                            params=params)
                pool = Pool(self.num_workers)
                with pool as p:
                    results_list = p.map(compute_track_predictions_partial, list(range(len(data))))
                pool.join()
                future_coords_batch = list(map(lambda x: x[0], results_list))
                timestamps_batch = list(map(lambda x: x[1], results_list))
                agent_ids_batch = list(map(lambda x: x[2], results_list))
                future_coords_offsets.extend(future_coords_batch)
                timestamps.extend(timestamps_batch)
                agent_ids.extend(agent_ids_batch)

            pred_path_temp = os.path.join(self.eval_base_path, 'pred_temp.csv')
            write_pred_csv(pred_path_temp,
                           timestamps=np.array(timestamps),
                           track_ids=np.array(agent_ids),
                           coords=np.concatenate(future_coords_offsets),
                           )
            metrics = compute_metrics_csv(self.eval_gt_path, pred_path_temp, [neg_multi_log_likelihood])
            # os.remove(pred_path_temp)
            return metrics['neg_multi_log_likelihood']

        return objective

    def onstep(self, res):
        with open(os.path.join(self.output_path, f'intermediate_parameters_{res.fun:.0f}.pkl'), 'wb') as f:
            pickle.dump(res.x, f)
        self.logger.info(f'Current iter: {self.counter} - Best neg loglikelihood: {res.fun:.3f} - Args: {res.x}')
        self.counter += 1

    def get_optimized_parameters(self):
        objective = self.make_objective()
        # checkpoint_saver = CheckpointSaver(f"./checkpoint_{self.experiment_name}.pkl", compress=9)
        gaussian_process_optimizer = gp_minimize(func=objective,
                                                 dimensions=self.dimensions,
                                                 acq_func='gp_hedge',
                                                 n_calls=self.n_optimizer_calls,
                                                 callback=[self.onstep],
                                                 random_state=42)
        best_parameters = gaussian_process_optimizer.x
        best_result = gaussian_process_optimizer.fun
        skopt_params = {param.name: value for param, value in zip(self.dimensions, best_parameters)}
        self.logger.info(f'Best Neg Log likelihood: {best_result},\ncorresponding parameters: {skopt_params}')
        with open(os.path.join(self.output_path, f'best_parameters_{best_result}.pkl'), 'wb') as f:
            pickle.dump(skopt_params, f)
        # preparation required to results dump to json
        for key, val in skopt_params.items():
            if type(val) == np.int64:
                skopt_params[key] = int(val)
        with open(os.path.join(self.output_path, f'best_parameters_{best_result}.json'), 'w') as f:
            json.dump(skopt_params, f)
        return skopt_params


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Bayesian (GP) optimization of negative log-likelihood of the ground truth data given the "
                     "multi-modal predictions, search for the best Kalman priors"))

    parser.add_argument(
        '--output-path-root',
        default='../output'
    )
    parser.add_argument(
        '--logging-path',
        default='../logs'
    )
    parser.add_argument(
        '--dataset-path',
        default='scenes/validate.zarr'
    )
    parser.add_argument(
        '--percentage-of-dataset-scenes',
        default=1.0,
        type=float
    )
    parser.add_argument(
        '--optimizer-calls-num',
        default=150,
        type=int
    )
    parser.add_argument(
        '--experiment-name',
        default=''
    )
    parser.add_argument(
        '--allocated-cpus-num',
        default=None,
        type=int
    )
    parser.add_argument(
        '--batch-size',
        default=None,
        type=int
    )
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    optimizer = KalmanInitializationOptimizer(args.output_path_root, args.logging_path,
                                              args.dataset_path,
                                              args.optimizer_calls_num,
                                              args.experiment_name,
                                              debug=args.debug,
                                              num_workers=args.allocated_cpus_num,
                                              batch_size=args.batch_size,
                                              percentage_of_dataset_scenes=args.percentage_of_dataset_scenes)
    optimizer.get_optimized_parameters()
