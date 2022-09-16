# -*- coding:utf-8 -*-

from typing import Iterator, Optional

import torch
from torch.utils.data.sampler import Sampler

__author__ = "mmuraoka"


class ContinuableSampler(Sampler):
    r"""Sampler that enables to continue the iteration in the loop while keeping the internal state.

    Parameters
    ----------
    sampler : Sampler
        base sampler to produce indices of the given dataset.
    resume_index : Optional[int]
        index to resume if any, by default, None (i.e., start from the beginning).
    generator_state : Optional[torch.Tensor]
        state for :class:`~torch.Generator`.
    """
    def __init__(self,
                 sampler: Sampler,
                 resume_index: Optional[int] = None,
                 generator_state: Optional["torch.Tensor"] = None):
        self.base_sampler = sampler
        self.resume_index = resume_index
        self.generator_state = generator_state
        generator = self._set_generator()
        if hasattr(self.base_sampler, "generator"):
            self.base_sampler.generator = generator

    def __iter__(self) -> Iterator[int]:
        # get the RNG state before executing random permutation
        self.generator_state = self.generator.get_state()

        lst_sample_indices = self.base_sampler.__iter__()
        continue_flag = False if self.resume_index is None else True
        for sample_idx in lst_sample_indices:
            if continue_flag:
                if sample_idx != self.resume_index:
                    continue
                else:
                    continue_flag = False
                    continue

            self.resume_index = sample_idx
            yield sample_idx

        # forget resume index because we don't need it anymore.
        self.resume_index = None

    def __len__(self) -> int:
        return len(self.base_sampler)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"base_sampler={self.base_sampler.__class__.__name__}, "
                f"resume_index={self.resume_index}, "
                f"generator={self.generator.__class__.__name__})")

    def _set_generator(self):
        generator = torch.Generator()
        if self.generator_state is not None:
            generator.set_state(self.generator_state.cpu())
        else:
            pass
        self.generator = generator
        return self.generator

    def get_resume_index(self):
        return self.resume_index

    def get_generator_state(self):
        return self.generator_state

    def set_epoch(self, epoch):
        if hasattr(self.base_sampler, "epoch"):
            _epoch = self.base_sampler.epoch
            if _epoch != epoch:
                self.base_sampler.set_epoch(epoch)
                # print(f"Set epoch to {epoch}")


if __name__ == "__main__":
    import os
    from pprint import pprint

    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.sampler import RandomSampler
    from torch.utils.data.distributed import DistributedSampler


    class ToyDataset(Dataset):
        def __init__(self, low, high):
            self.samples = list(range(low, high))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    print(f"ContinuableRandomSampler")

    batch_size = 4
    dataset = ToyDataset(low=10, high=30)
    base_sampler = RandomSampler(dataset)
    sampler = ContinuableSampler(base_sampler)
    # Keep `num_workers=0` and do not use `num_workers>0`
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    print(sampler)

    print(f"- First run: save random state of the sampler")
    sampler_state_file = "sampler_state.pt"
    resume_bidx = 1
    gold = []  # for testing
    for idx, batch in enumerate(data_loader):
        print(idx, batch)
        if idx == resume_bidx:
            resume_index = data_loader.sampler.get_resume_index()
            generator_state = data_loader.sampler.get_generator_state()
            print(f"batch idx: {idx}, you can resume from here")
            sampler_state = {"resume_index": resume_index,
                             "generator_state": generator_state}
            print(sampler_state)
            torch.save(sampler_state, sampler_state_file)
        elif idx > resume_bidx:
            gold.append(batch.tolist())

    print(f"Clear the state.")
    del dataset
    del sampler
    del data_loader
    del resume_index
    del generator_state

    print(f"- Second run: resuming from the saved state")
    loaded_state = torch.load(sampler_state_file)
    print(loaded_state)
    dataset = ToyDataset(low=10, high=30)
    base_sampler = RandomSampler(dataset)
    sampler = ContinuableSampler(base_sampler,
                                 resume_index=loaded_state["resume_index"],
                                 generator_state=loaded_state["generator_state"])
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)

    pred = []
    for idx, batch in enumerate(data_loader):
        print(idx, batch)
        pred.append(batch.tolist())

    # Check if the instances processed after saving/restoring the random state
    assert gold == pred
    del gold
    del pred
    os.remove(sampler_state_file)
    print(f"[Test - RandomSampler] pass.")

    print("\n" + "-"*10 + "\n")

    print(f"(Pseudo) DistributedContinuableRandomSampler")

    dataset = ToyDataset(low=10, high=30)
    batch_size = 2
    world_size = 4
    data_loaders = []
    for rank in range(world_size):
        base_sampler = DistributedSampler(dataset=dataset,
                                          num_replicas=world_size,
                                          rank=rank,
                                          shuffle=True)
        sampler = ContinuableSampler(base_sampler)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        data_loaders.append(data_loader)

    print("- First run")
    resume_bidx = 1
    gold, output = [], []
    sampler_states = []
    for rank, data_loader in enumerate(data_loaders):
        print(f"rank = {rank}")
        for idx, batch in enumerate(data_loader):
            print(batch)
            output.extend(batch.tolist())
            if idx == resume_bidx:
                resume_index = data_loader.sampler.get_resume_index()
                generator_state = data_loader.sampler.get_generator_state()
                sampler_state = {"resume_index": resume_index,
                                 "generator_state": generator_state}
                sampler_states.append(sampler_state)
                print(f"batch idx: {idx}, you can resume from here")
                print(sampler_state)
            elif idx > resume_bidx:
                gold.append(batch.tolist())

    assert dataset.samples == sorted(output)

    sampler_state_file = "sampler_states.pt"
    torch.save(sampler_states, sampler_state_file)

    print(f"Clear the state.")
    del dataset
    del sampler
    del data_loader
    del data_loaders
    del resume_index
    del generator_state
    del sampler_state
    del sampler_states

    print(f"- Second run")
    loaded_states = torch.load(sampler_state_file)
    pprint(loaded_states)

    dataset = ToyDataset(low=10, high=30)
    data_loaders = []
    for rank in range(world_size):
        sampler_state = loaded_states[rank]
        base_sampler = DistributedSampler(dataset=dataset,
                                          num_replicas=world_size,
                                          rank=rank,
                                          shuffle=True)
        sampler = ContinuableSampler(base_sampler,
                                     resume_index=sampler_state["resume_index"],
                                     generator_state=sampler_state["generator_state"])
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        data_loaders.append(data_loader)

    pred = []
    for rank, data_loader in enumerate(data_loaders):
        print(f"rank = {rank}")
        for idx, batch in enumerate(data_loader):
            print(batch)
            pred.append(batch.tolist())

    # Check if the instances processed after saving/restoring the random state
    assert gold == pred
    print(f"[Test - DistributedSampler] pass.")
    os.remove(sampler_state_file)