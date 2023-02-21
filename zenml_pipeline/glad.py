from zenml.pipelines import pipeline
#Define the pipeline
@pipeline
def glad_pipeline(
    load_dataset,
    create_chunks,
    train,
    process_results,
):
    #create a flow of the pipeline using the steps. \
    #If a step's input parameter is an output from another step then that step is only executed after the previous step execution is completed.
    dataset, model_configurations = load_dataset()
    chunks = create_chunks(model_configurations, dataset)
    result, _ = train(model_configurations, chunks)
    #If a step's input parameter is not dependent on another steps's out then that step is run in parallel.
    assessment_result = process_results(result)