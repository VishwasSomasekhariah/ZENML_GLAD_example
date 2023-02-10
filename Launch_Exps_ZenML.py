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
from zenml_steps import (
    load_dataset,
    create_chunks,
    train,
    process_results,
    LoadParameters,
)
from zenml_pipeline import glad_pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", dest="config_file", default="config_OCPool.yml"
    )
    parser.add_argument("--dataset-name", dest="dataset_name", default="dd")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = "config_files/" + args.config_file

    glad_pipeline_instance = glad_pipeline(
        load_dataset=load_dataset(LoadParameters(config_file=config_file)),
        create_chunks=create_chunks(),
        train=train(),
        process_results=process_results(),
    )

    glad_pipeline_instance.run()
