import os
from typing import Type

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
import pickle
from loader.GraphDataClass import TUDatasetManager

class TUDatasetManagerMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (TUDatasetManager,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[TUDatasetManager]) -> TUDatasetManager:
        """Read from artifact store"""
        super().load(data_type)
        with fileio.open(os.path.join(self.uri, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data

    def save(self, my_obj: TUDatasetManager) -> None:
        """Write to artifact store"""
        super().save(my_obj)
        with fileio.open(os.path.join(self.uri, 'data.pkl'), 'wb') as f:
            pickle.dump(my_obj, f)