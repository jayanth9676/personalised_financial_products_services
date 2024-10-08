import sagemaker
from sagemaker.xgboost import XGBoost
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Assuming your data is already in S3
train_data = 's3://your-bucket/train/train.csv'
validation_data = 's3://your-bucket/validation/validation.csv'

xgb = XGBoost(
    entry_point='xgboost_script.py',
    framework_version='1.0-1',
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
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://your-bucket/output'
)

xgb.fit({'train': train_data, 'validation': validation_data})

predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Save the endpoint name for use in Lambda functions
print(f"SageMaker Endpoint: {predictor.endpoint_name}")