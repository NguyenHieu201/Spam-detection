from typing import Iterable, Optional, Sequence, Union
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t


class SiameseDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: _collate_fn_t | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def __len__(self) -> int:
        return super().__len__()
