import os
from typing import Type

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from config.base import Grid
import pickle

class GridMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Grid,) #This is how ZenML knows to use this materializer type when it encounters Grid in any of the steps input or output parameter.
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Grid]) -> Grid:
        """Read from artifact store"""
        super().load(data_type)
        with fileio.open(os.path.join(self.uri, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data

    def save(self, my_obj: Grid) -> None:
        """Write to artifact store"""
        super().save(my_obj)
        with fileio.open(os.path.join(self.uri, 'data.pkl'), 'wb') as f:
            pickle.dump(my_obj, f)