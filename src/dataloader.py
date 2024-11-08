import torch

def create_train_loader(train_dataset, batch_size, args):
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
             batch_size=batch_size,
             shuffle=(train_sampler is None),
             sampler=train_sampler,
             num_workers=args.workers,
             pin_memory=True)
    return train_loader

def create_eval_loader(eval_dataset, batch_size, args):
    if args.distributed:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    else:
        eval_sampler = None
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=args.workers,
        pin_memory=True)
    return eval_loader