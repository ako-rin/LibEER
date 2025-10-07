import torch
from torch.utils.data import DataLoader,RandomSampler, SequentialSampler
from tqdm import tqdm

from ..utils.metric import Metric
from ..utils.store import save_state

def train(model, dataset_train, dataset_val, dataset_test, device, output_dir="result/", metrics=None, metric_choose=None, optimizer=None, scheduler=None, warmup_scheduler=None, batch_size=16, epochs=40, criterion=None, loss_func=None, loss_param=None):
    """
    训练+验证+测试统一入口。

    Args:
        model: 已构建好的 PyTorch 模型。
        dataset_train, dataset_val, dataset_test: PyTorch Dataset 对象。
        device: 训练设备（如 'cuda' 或 torch.device）。
        output_dir (str): 最优模型的保存目录。
        metrics (list[str] | str | None): 评估指标列表或逗号分隔字符串；支持：
            - 'acc'（准确率）
            - 'macro-f1'
            - 'micro-f1'
            - 'weighted-f1'
            - 'ck'（Cohen's kappa）
            为空则默认 ['acc']。
        metric_choose (str | None): 作为“最佳模型”判定的指标名；默认取 metrics[0]。
        optimizer, scheduler, warmup_scheduler: 优化器、学习率调度器与 warmup 调度器。
        batch_size (int): 训练/验证/测试的 batch 大小。
        epochs (int): 训练轮数。
        criterion: 监督损失（如交叉熵/标签平滑 CE 等）。签名应为 criterion(logits, targets)。
        loss_func (callable | None): 可选的辅助损失函数；签名为 loss_func(loss_param)->scalar。
        loss_param (Any): 传给 loss_func 的参数（如权重与参数组等）。

    Returns:
        dict: 在测试集上的各指标分数，如 {'acc': 0.93, 'macro-f1': 0.91, ...}

    Notes:
        - 训练与验证阶段的 loss 都会加上辅助损失：loss = criterion(...) + (loss_func(loss_param) if loss_func else 0)。
        - 最佳模型根据 metric_choose 在验证集上选择，并保存到 output_dir。
    """
    if metrics is None:
        metrics = ['acc']
    if metric_choose is None:
        metric_choose = metrics[0]
    # data sampler for train and test data
    sampler_train = RandomSampler(dataset_train)
    sampler_val = RandomSampler(dataset_val)  # Use RandomSampler for val to match PGCN paper
    sampler_test = SequentialSampler(dataset_test)
    # load dataset (add shuffle=True and drop_last=True for training as per PGCN paper)
    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size, num_workers=4, 
        shuffle=False, drop_last=True  # shuffle via sampler, drop_last=True to ensure batch size consistency
    )
    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val, batch_size=batch_size, num_workers=4, 
        shuffle=False, drop_last=False  # shuffle via sampler
    )
    data_loader_test = DataLoader(
        dataset_test, sampler=sampler_test, batch_size=batch_size, num_workers=4,
        shuffle=False  # sequential order for test
    )
    model = model.to(device)
    best_metric = {s: 0. for s in metrics}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # create Metric object
        metric = Metric(metrics)
        train_bar = tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=
        f"Train Epoch {epoch}/{epochs}: lr:{optimizer.param_groups[0]['lr']}")
        for idx, (samples, targets) in train_bar:
            # load the samples into the device
            samples = samples.to(device)
            targets = targets.to(device)
            
            # Handle targets: convert one-hot to class indices if needed
            if targets.dim() > 1 and targets.size(-1) > 1:
                # One-hot encoded: (batch_size, num_classes) -> (batch_size,)
                targets = torch.argmax(targets, dim=-1)
            elif targets.dim() > 1:
                # (batch_size, 1) -> (batch_size,)
                targets = targets.squeeze(-1)
            
            # Ensure targets are integer class indices
            if targets.dtype != torch.long:
                targets = targets.long()
            
            optimizer.zero_grad()
            # perform emotion recognition
            outputs = model(samples)
            # some models (e.g., PGCN) return (logits, laplacian, feature); extract logits
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            # calculate the loss value
            loss = criterion(logits, targets) +  (0 if loss_func is None else loss_func(loss_param))
            metric.update(torch.argmax(logits, dim=1), targets, loss.item())
            train_bar.set_postfix_str(f"loss: {loss.item():.2f}")

            loss.backward()
            optimizer.step()
            
            # Warmup scheduler dampening (PGCN paper style)
            if warmup_scheduler is not None and idx < len(data_loader_train) - 1:
                with warmup_scheduler.dampening():
                    pass

        # Step warmup scheduler after each epoch (except last batch which is handled above)
        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if scheduler is not None:
                    scheduler.step()
        elif scheduler is not None:
            scheduler.step()
        print("\033[32m train state: " + metric.value())
        metric_value = evaluate(model, data_loader_val, device, metrics, criterion, loss_func, loss_param)
        for m in metrics:
            # if metric is the best, save the model state
            if metric_value[m] > best_metric[m]:
                best_metric[m] = metric_value[m]
                save_state(output_dir, model, optimizer, epoch+1, metric=m)
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-best{metric_choose}")['model'])
    metric_value = evaluate(model, data_loader_test, device, metrics, criterion, loss_func, loss_param)
    for m in metrics:
        print(f"best_val_{m}: {best_metric[m]:.2f}")
        print(f"best_test_{m}: {metric_value[m]:.2f}")
    return metric_value

@torch.no_grad()
def evaluate(model, data_loader, device, metrics, criterion, loss_func, loss_param):
    """
    在给定 DataLoader 上评估模型，返回指标字典。

    Args:
        model: 已在正确设备上的模型。
        data_loader: PyTorch DataLoader。
        device: 设备。
        metrics (list[str]): 评估指标，同 train 中说明。
        criterion: 监督损失函数（criterion(logits, targets)）。
        loss_func (callable | None): 辅助损失函数（loss_func(loss_param)）。
        loss_param: 传入辅助损失的参数。

    Returns:
        dict: 评估得到的指标分数字典。
    """
    model.eval()
    # create Metric object
    metric = Metric(metrics)
    for idx, (samples, targets) in tqdm(enumerate(data_loader), total=len(data_loader),
                                        desc=f"Evaluating : "):
        # load the samples into the device
        samples = samples.to(device)
        targets = targets.to(device)
        
        # Handle targets: convert one-hot to class indices if needed
        if targets.dim() > 1 and targets.size(-1) > 1:
            # One-hot encoded: (batch_size, num_classes) -> (batch_size,)
            targets = torch.argmax(targets, dim=-1)
        elif targets.dim() > 1:
            # (batch_size, 1) -> (batch_size,)
            targets = targets.squeeze(-1)
        
        # Ensure targets are integer class indices
        if targets.dtype != torch.long:
            targets = targets.long()

        # perform emotion recognition
        outputs = model(samples)
        # some models (e.g., PGCN) return (logits, laplacian, feature); extract logits
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # calculate the loss value
        loss = criterion(logits, targets) + (0 if loss_func is None else loss_func(loss_param))
        # one hot code
        # loss = criterion(outputs, targets)
        metric.update(torch.argmax(logits, dim=1), targets, loss.item())

    print("\033[34m eval state: " + metric.value())
    return metric.values