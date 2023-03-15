from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

docker_settings = DockerSettings(
    required_integrations=[MLFLOW], requirements="../requirements.txt"
)


@pipeline(settings={"docker": docker_settings})
def glad_pipeline(
    load_dataset,
    create_chunks,
    train,
    process_results,
):
    dataset, model_configurations = load_dataset()
    chunks = create_chunks(model_configurations, dataset)
    result, _ = train(model_configurations, chunks)
    assessment_result = process_results(result)
