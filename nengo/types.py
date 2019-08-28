from typing import (
    Any,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    runtime_checkable,
    Union,
)

import numpy as np

from nengo.base import NengoObject
from nengo.network import Network


@runtime_checkable
class SimulatorProtocol(Protocol):
    data: Mapping[NengoObject, Union[List[np.ndarray], NamedTuple]]
    # max_steps: int  # Only exists for Nengo SpiNNaker

    def __init__(
        self,
        network: Network,
        dt: float = 0.001,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        ...

    def close(self) -> None:
        ...

    def run(self, time_in_seconds: float, progress_bar: Optional[Any] = None) -> None:
        ...

    def run_steps(self, steps: int, progress_bar: Optional[Any] = None) -> None:
        ...

    def step(self) -> None:
        ...
