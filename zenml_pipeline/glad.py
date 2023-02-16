from zenml.pipelines import pipeline

@pipeline
def glad_pipeline(
    load_dataset,
    create_chunks,
    train,
    process_results,
):
    dataset, model_configurations = load_dataset()
    chunks = create_chunks(model_configurations, dataset)
    result = train(model_configurations, chunks)
    assessment_result = process_results(result)