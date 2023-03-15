# Raising the Bar in Graph-level Anomaly Detection (GLAD)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from zenml_steps.all_steps import (
    load_dataset,
    create_chunks,
    train,
    process_results,
    LoadParameters,
)
from zenml_pipeline.glad import glad_pipeline
from zenml_materializers.grid_materializer import GridMaterializer
from zenml_materializers.tu_materializer import TUDatasetManagerMaterializer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", dest="config_file", default="config_OCGTL.yml"
    )
    parser.add_argument("--dataset-name", dest="dataset_name", default="dd")
    return parser.parse_args()


if __name__ == "__main__":
    #Get command line arguments
    args = get_args()
    config_file = "config_files/" + args.config_file

    #Initialize the pipeline instance
    glad_pipeline_instance = glad_pipeline(
        load_dataset=load_dataset(LoadParameters(config_file=config_file)),
        create_chunks=create_chunks(),
        train=train(),
        process_results=process_results(),
    )

    #Run the pipeline instance
    glad_pipeline_instance.run()

# # Raising the Bar in Graph-level Anomaly Detection (GLAD)
# # Copyright (c) 2022 Robert Bosch GmbH
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU Affero General Public License as published
# # by the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU Affero General Public License for more details.
# #
# # You should have received a copy of the GNU Affero General Public License
# # along with this program.  If not, see <https://www.gnu.org/licenses/>.
# #
# import random
# from typing import Any, Dict, List
# import torch
# import numpy as np
# import argparse
# from config.base import Grid, Config
# from evaluation.Experiments import runGraphExperiment
# from evaluation.Kfolds_Eval import KFoldEval
# from zenml.steps import step, BaseParameters
# from zenml.pipelines import pipeline


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config-file", dest="config_file", default="config_OCPool.yml"
#     )
#     parser.add_argument("--dataset-name", dest="dataset_name", default="dd")
#     return parser.parse_args()


# class LoadParameters(BaseParameters):
#     """Parameters for the load step"""

#     config_file: str = "config_OCPool.yml"
#     dataset_name: str = "dd"


# @pipeline
# def glad_pipeline(
#     load_dataset,
#     create_chunks,
#     train,
#     process_results,
# ):
#     dataset, model_configurations = load_dataset()
#     chunks = create_chunks(model_configurations, dataset)
#     result = train(model_configurations, chunks)
#     assessment_result = process_results(result)


# # config file params and dataset name can be passed as BaseParameters to the step function
# @step
# def load_dataset(params: LoadParameters):
#     model_configurations = Grid(params.config_file, params.dataset_name)
#     model_configuration = Config(**model_configurations[0])
#     dataset = model_configuration.dataset
#     return dataset, model_configurations


# class FoldsParameters(BaseParameters):
#     """Parameters for the chunks step"""

#     num_folds: int = 5


# @step
# def create_chunks(
#     model_configurations: Grid, dataset, params: FoldsParameters
# ) -> Dict[str, Dict[str, List[Any]]]:
#     # create folds
#     chunks = {}
#     for fold_k in range(params.num_folds):
#         num_repeat = model_configurations[0]["num_repeat"]
#         chunks[fold_k] = {}
#         for cls in dataset.num_cls:
#             chunks[fold_k][cls] = []
#             for i in range(num_repeat):
#                 torch.backends.cudnn.deterministic = True
#                 torch.backends.cudnn.benchmark = True
#                 np.random.seed(i + 40)
#                 random.seed(i + 40)
#                 torch.manual_seed(i + 40)
#                 torch.cuda.manual_seed(i + 40)
#                 torch.cuda.manual_seed_all(i + 40)
#                 train_data, val_data = dataset.get_model_selection_fold(fold_k, cls)
#                 test_data = dataset.get_test_fold(fold_k)
#                 # add data to dict per class
#                 chunks[fold_k][cls].append([train_data, val_data, test_data])
#     return chunks


# @step
# # take the required step configurations
# def train(model_configurations: Grid, folds: Dict[str, Dict[str, List[Any]]]):
#     exp_class = runGraphExperiment
#     best_config = model_configurations[0]
#     experiment = exp_class(best_config)
#     result = {}
#     saved_scores = {}
#     for fold in folds.keys():
#         classes = folds[fold]
#         val_auc_list, test_auc_list, test_f1_list = [], [], []
#         saved_results = {}
#         saved_scores[fold] = {}
#         result[fold] = {}
#         for cls in classes.keys():
#             data = classes[cls]
#             for i in range(len(data)):
#                 (
#                     val_auc,
#                     test_auc,
#                     test_ap,
#                     test_f1,
#                     scores,
#                     labels,
#                 ) = experiment.run_test(
#                     [data[i][0], data[i][1], data[i][2]],
#                     cls,
#                 )
#                 saved_results["scores_" + str(i)] = scores.tolist()
#                 saved_results["labels_" + str(i)] = labels.tolist()

#                 val_auc_list.append(val_auc)
#                 test_auc_list.append(test_auc)
#                 test_f1_list.append(test_f1)

#             val_auc = sum(val_auc_list) / len(data)
#             test_auc = sum(test_auc_list) / len(data)
#             test_f1 = sum(test_f1_list) / len(data)

#             if best_config["save_scores"]:
#                 saved_scores[fold][cls] = saved_results

#             result[fold][cls] = {
#                 "best_config": best_config,
#                 "VAL_auc_" + str(cls): val_auc,
#                 "TS_auc_" + str(cls): test_auc,
#                 "TS_f1_" + str(cls): test_f1,
#             }

#     return result, saved_scores


# @step
# def process_results(result):
#     # read from the results object, do some processing and write to assessment object
#     TS_aucs, TS_f1s, = (
#         [],
#         [],
#     )
#     assessment_results = {}

#     for normal_cls in range(len(result[0])):
#         Fold_TS_aucs, Fold_TS_f1s = [], []

#         for i in range(len(result)):
#             try:
#                 variant_scores = result[i][normal_cls]
#                 # Fold_VAL_aucs.append(variant_scores['VAL_auc_'+str(normal_cls)])
#                 Fold_TS_aucs.append(variant_scores["TS_auc_" + str(normal_cls)])
#                 Fold_TS_f1s.append(variant_scores["TS_f1_" + str(normal_cls)])

#             except Exception as e:
#                 print(e)
#         # Fold_VAL_aucs = np.array(Fold_VAL_aucs)
#         Fold_TS_aucs = np.array(Fold_TS_aucs)
#         Fold_TS_f1s = np.array(Fold_TS_f1s)

#         # results['avg_VAL_auc_' + str(normal_cls)] = Fold_VAL_aucs.mean()
#         # results['std_VAL_auc_' + str(normal_cls)] = Fold_VAL_aucs.std()
#         assessment_results["avg_TS_auc_" + str(normal_cls)] = Fold_TS_aucs.mean()
#         assessment_results["std_TS_auc_" + str(normal_cls)] = Fold_TS_aucs.std()
#         assessment_results["avg_TS_f1_" + str(normal_cls)] = Fold_TS_f1s.mean()
#         assessment_results["std_TS_f1_" + str(normal_cls)] = Fold_TS_f1s.std()

#         # VAL_aucs.append(Fold_VAL_aucs)
#         TS_aucs.append(Fold_TS_aucs)
#         TS_f1s.append(Fold_TS_f1s)

#     # VAL_aucs = np.array(VAL_aucs)
#     TS_aucs = np.array(TS_aucs)
#     TS_f1s = np.array(TS_f1s)

#     # avg_VAL_auc = np.mean(VAL_aucs,0)
#     avg_TS_auc = np.mean(TS_aucs, 0)
#     avg_TS_f1 = np.mean(TS_f1s, 0)

#     # results['avg_VAL_auc'] = avg_VAL_auc.mean()
#     # results['std_VAL_auc'] = avg_VAL_auc.std()
#     assessment_results["avg_TS_auc"] = avg_TS_auc.mean()
#     assessment_results["std_TS_auc"] = avg_TS_auc.std()
#     assessment_results["avg_TS_f1"] = avg_TS_f1.mean()
#     assessment_results["std_TS_f1"] = avg_TS_f1.std()

#     return assessment_results


# if __name__ == "__main__":
#     args = get_args()
#     config_file = "config_files/" + args.config_file

#     glad_pipeline_instance = glad_pipeline(
#         load_dataset = load_dataset(LoadParameters(config_file=config_file)),
#         create_chunks = create_chunks(),
#         train = train(),
#         process_results = process_results(),
#     )

#     glad_pipeline_instance.run()