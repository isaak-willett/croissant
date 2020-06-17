from pathlib import Path
import json
import logging
import tempfile

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, GroupKFold, LeaveOneOut,
                                     ShuffleSplit, GridSearchCV,
                                     StratifiedKFold, ParameterGrid)
import pandas as pd
import mlflow
import argschema

from croissant.schemas import (TrainingException, TrainingSchema)
from croissant.features import (Roi, RoiMetadata, FeatureExtractor,
                                feature_pipeline)

default_data = {
    'training_data': (Path(__file__).parents[1] / 'test' / 'resources'/ 'dev_train_rois.json').as_posix(),
    'cv_strategy': 'k_fold',
    'cv_kwargs': ['n_splits', '5']
}

generator_mapping = {
    'k_fold': KFold,
    'loo': LeaveOneOut,
    'group_kfold': GroupKFold,
    'shuffle_split': ShuffleSplit,
    'stratified_k_fold': StratifiedKFold
}

logger = logging.getLogger('TrainClassifier')


def train_classifier(environment: str, experiment_name:str,
                     training_data: Path, output_dir: Path,
                     cv_strategy: str, cv_kwargs: dict,
                     search_grid: dict):
    # set tracker
    if 'dev' or 'test' in environment:
        tracker = LocalTracker(output_dir)
    else:
        tracker = mlflow
        tracker.start_run(run_name=experiment_name)

    # load the data, assume json format
    with open(training_data.as_posix(), 'r') as open_training:
        training_data_loaded = json.load(open_training)
        logger.info("Loaded ROI data from manifest.")

    rois = []
    dff_traces = []
    metadatas = []
    labels = []

    # extract data
    logger.info("Extracting ROI data from manifest data")
    for roi_data in training_data_loaded:
        dff_traces.append(roi_data['trace'])
        roi = Roi(roi_id=roi_data['roi_id'], coo_cols=roi_data['coo_cols'],
                  coo_rows=roi_data['coo_rows'], coo_data=roi_data['coo_data'],
                  image_shape=roi_data['image_shape'])
        roi_meta = RoiMetadata(depth=roi_data['depth'],
                               full_genotype=roi_data['full_genotype'],
                               targeted_structure=roi_data['targeted_structure'],
                               rig=roi_data['rig'])
        labels.append(roi_data['label'])
        rois.append(roi)
        metadatas.append(roi_meta)
    logger.info("Extracted all ROI data and formatted for feature extraction.")

    logger.info('Extracting features!')
    feature_extractor = FeatureExtractor(rois=rois,
                                         dff_traces=dff_traces,
                                         metadata=metadatas)
    features = feature_extractor.run()
    logger.info('Feature extraction complete!')

    logger.info('Logging features to tracker!')
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        feature_path = Path(tmp_dir) / 'Features.h5'
        features.to_hdf(feature_path, key="Features", mode='w')
        # tracker.log_artifact(feature_path)

    # Fitting model
    logger.info('Fitting model to data!')
    pipeline = feature_pipeline()
    model = LogisticRegression(penalty='elasticnet', solver='saga')
    pipeline.steps.append(('model', model))

    scorers = {'AUC': 'roc_auc',
               'Accuracy': 'accuracy'}

    if cv_strategy:
        logger.info(f"Strategy selected: {cv_strategy}, grid search will "
                    f"occur with the selected cross validation strategy on "
                    f"the specified parameter grid.")
        # grid search with cross validation
        selected_generator = generator_mapping[cv_strategy]
        selected_generator = selected_generator(**cv_kwargs)
        gs = GridSearchCV(pipeline, param_grid=search_grid,
                          scoring=scorers, cv=selected_generator,
                          refit='AUC')
        fitted_model = gs.fit(features, labels)
        score_stat_dict = {}
        for score_key in scorers.keys():
            score_stat_dict[f'Mean {score_key}'] = gs.cv_results_[
                f'mean_test_{score_key}']
            score_stat_dict[f'Std {score_key}'] = gs.cv_results_[
                f'std_test_{score_key}']
        score_stat_dict['Params'] = gs.cv_results_['params']
        score_stat_frame = pd.DataFrame.from_dict(score_stat_dict)

    else:
        logger.info("No CV strategy selected, the grid search will occur "
                    "with no cross validation.")
        best_score = 0
        # grid search without cross validation
        scores = []
        grid_points = []
        for g in ParameterGrid(search_grid):
            pipeline.set_params(**g)
            pipeline.fit(features, labels)
            score = pipeline.score(features, labels)
            scores.append(score)
            grid_points.append(g)
            if score > best_score:
                fitted_model = pipeline
                fitted_model.best_score_ = score
                fitted_model.best_params_ = g
        score_stat_frame = pd.DataFrame.from_dict({'Score': scores,
                                                   'Params': grid_points})

    logger.info(f"Model fitted, the best score is {fitted_model.best_score_} "
                f"and the best parameters are {fitted_model.best_params_}.")

    logger.info("Logging classification metrics")
    tracker.log_metrics(score_stat_frame)


class LocalTracker:

    def __init__(self, output_dir):
        pass


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):

        # set up logger
        logger.setLevel(self.args.pop('log_level'))

        # prepare args for handoff
        self.args['training_data'] = Path(self.args['training_data'])
        self.args['output_dir'] = Path(self.args['output_dir'])
        if self.args['cv_strategy']:
            i = iter(self.args['cv_kwargs'])
            b = dict(zip(i, i))
            self.args['cv_kwargs'] = {key: int(value) for key,
                                      value in b.items()}
        search_grid_path = self.args.pop('search_grid_path')
        if search_grid_path:
            with open(search_grid_path) as open_grid:
                search_grid = json.load(open_grid)
        else:
            search_grid = {'model__l1_ratio': [0.25, 0.5, 0.75]}

        train_classifier(search_grid=search_grid, **self.args)


if __name__ == '__main__':
    trainer = ClassifierTrainer()
    trainer.train()
