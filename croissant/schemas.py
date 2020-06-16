import argschema
import marshmallow as mm


class TrainingException(Exception):
    pass


class TrainingSchema(argschema.ArgSchema):
    environment = argschema.fields.String(
        required=False,
        default='dev',
        validator=mm.validate.OneOf(['dev', 'test', 'prod']),
        description=("Either 'dev', 'prod', or 'test'"))
    experiment_name = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description=("Experiment name (for organization in MLFlow)"))
    training_data = argschema.fields.InputFile(
        required=True,
        description=("Input file in json format containing the ROIs "
                    "with which to build the classifier."))
    output_dir = argschema.fields.OutputDir(
        required=False,
        default='.',
        description=("Where to save output from dev runs (tracking metrics and "
                    "artifacts). Ignored if not in dev environment. Defaults "
                    "to current directory."))
    cv_strategy = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description=(" Strategy for crossvalidation. One of 'k_fold', 'loo', "
                    "'group_kfol' 'shuffle_split' – corresponding to "
                    "crossvalidator classes in Scikit-Learn. If None, "
                    "crossvalidation will be skipped."))
    cv_kwargs = argschema.fields.List(
        argschema.fields.String(),
        required=False,
        default=None,
        allow_none=True,
        description=("List of key-value pairs, corresponding to keyword "
                    "arguments to pass to the crossvalidator constructor. For "
                    "'groupkfold' need to also include `groups <name of "
                    "grouping column in input data>`."))
