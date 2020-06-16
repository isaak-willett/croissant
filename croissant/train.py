from pathlib import Path
import json
import logging

import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import mlflow
import argschema

from croissant.schemas import (TrainingException, TrainingSchema)
from croissant.features import (Roi, RoiMetadata, FeatureExtractor,
                                feature_pipeline)

default_data = {
    'training_data': (Path(__file__).parents[1] / 'test' / 'resources'/ 'dev_train_rois.json').as_posix()
}

cv_mapping = {
    ''
}

logger = logging.getLogger('TrainClassifier')


def train_classifier(environment: str, experiment_name:str,
                     training_data: Path, output_dir: Path,
                     cv_strategy: str, cv_kwargs: dict):
    # set tracker
    if 'dev' or 'test' in environment:
        tracker = LocalTracker()
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

    logger.info("Logging feature artifacts")
    '''
    output_file_path = output_dir / f"classification_features.h5"
    with h5py.File(output_file_path, 'w') as open_output:
        open_output['rois'] = rois
        open_output['rois_metadata'] = metadatas
        open_output['dff_traces'] = dff_traces
    
    '''
    logger.info('Extracting features!')
    feature_extractor = FeatureExtractor(rois=rois,
                                         dff_traces=dff_traces,
                                         metadata=metadatas)
    features = feature_extractor.run()
    logger.info('Feature extraction complete!')

    pipeline = feature_pipeline()
    regressor = LogisticRegression(penalty='elasticnet', solver='saga',
                                   l1_ratio=0.5)
    pipeline.steps.append(('regr', regressor))

    if cv_strategy:
        scores = cross_val_score(pipeline, features, labels, cv=cv_strategy,
                                 scoring='roc_auc')
    else:
        pipeline.fit(X=features, y=labels)
        pipeline.score(X=features, y=labels)


class LocalTracker:

    def __init__(self):
        pass


class ClassifierTrainer(argschema.ArgSchemaParser):
    default_schema = TrainingSchema

    def train(self):

        # set up logger
        logger.setLevel(self.args.pop('log_level'))

        # prepare args for handoff
        self.args['training_data'] = Path(self.args['training_data'])
        self.args['output_dir'] = Path(self.args['output_dir'])
        train_classifier(**self.args)


if __name__ == '__main__':
    trainer = ClassifierTrainer(input_data=default_data)
    trainer.train()