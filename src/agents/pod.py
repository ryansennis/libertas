import mesa

from .worker import Worker

class Pod(mesa.Agent):
    def __init__(
        self: 'Pod',
        model: mesa.Model,
        workers: list[Worker],
        *args,
        **kwargs
    ) -> None:
        super().__init__(model, *args, **kwargs)

        self.workers = workers