import boto3
import sagemaker
from sagemaker.xgboost import XGBoost
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

def train_and_deploy_model(
    role,
    bucket,
    train_key,
    validation_key,
    output_path,
    instance_type='ml.m5.xlarge',
    instance_count=1
):
    sagemaker_session = sagemaker.Session()
    
    # Define the XGBoost estimator
    xgb = XGBoost(
        entry_point='xgboost_script.py',
        framework_version='1.5-1',
        hyperparameters={
            'max_depth': 5,
            'eta': 0.2,
            'gamma': 4,
            'min_child_weight': 6,
            'subsample': 0.8,
            'objective': 'binary:logistic',
            'num_round': 100
        },
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=output_path
    )
    
    # Define hyperparameter ranges for tuning
    hyperparameter_ranges = {
        'max_depth': IntegerParameter(3, 10),
        'eta': ContinuousParameter(0.01, 0.5),
        'gamma': ContinuousParameter(0, 10),
        'min_child_weight': IntegerParameter(1, 10),
        'subsample': ContinuousParameter(0.5, 1.0)
    }
    
    # Create the hyperparameter tuner
    tuner = HyperparameterTuner(
        xgb,
        objective_metric_name='validation:auc',
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=20,
        max_parallel_jobs=3
    )
    
    # Start the hyperparameter tuning job
    tuner.fit({
        'train': sagemaker.TrainingInput(f's3://{bucket}/{train_key}', content_type='csv'),
        'validation': sagemaker.TrainingInput(f's3://{bucket}/{validation_key}', content_type='csv')
    })
    
    # Deploy the best model
    predictor = tuner.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium'
    )
    
    return predictor.endpoint_name

if __name__ == '__main__':
    role = 'your-sagemaker-role-arn'
    bucket = 'your-s3-bucket'
    train_key = 'path/to/train/data.csv'
    validation_key = 'path/to/validation/data.csv'
    output_path = 's3://your-s3-bucket/model/output'
    
    endpoint_name = train_and_deploy_model(role, bucket, train_key, validation_key, output_path)
    print(f"Model deployed to endpoint: {endpoint_name}")