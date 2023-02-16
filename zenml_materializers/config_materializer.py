import os
from typing import Type

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from config.base import Config
import pickle

class ConfigMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Config,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Config]) -> Config:
        """Read from artifact store"""
        super().load(data_type)
        with fileio.open(os.path.join(self.uri, 'data.pkl'), 'r') as f:
            data = pickle.load(f)
        return data

    def save(self, my_obj: Config) -> None:
        """Write to artifact store"""
        super().save(my_obj)
        with fileio.open(os.path.join(self.uri, 'data.pkl'), 'w') as f:
            pickle.dump(my_obj, f)