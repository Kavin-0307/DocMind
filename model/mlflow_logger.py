import mlflow



def log_experiment(model_name,params,metrics):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        for key,value in metrics.items():
            mlflow.log_metric(key,value)
        print(f"Experiment logged to MLFlow:{model_name}")
    