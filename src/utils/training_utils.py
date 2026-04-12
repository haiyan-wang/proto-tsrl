from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F



Tensor = torch.Tensor
Batch = Union[Dict[str, Any], Tuple[Any, ...], List[Any]]

# ============================================================
# ============================================================

def _unwrap_model(
        model: nn.Module
    ) -> nn.Module:
    """
    Return the underlying module if model is wrapped in DataParallel / DDP.
    """

    return model.module if hasattr(model, "module") else model

def _move_to_device(
        x: Any, device: 
        torch.device
    ) -> Any:

    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    
    return x

# ============================================================
# ============================================================

def _extended_forward(
        model: nn.Module, 
        x: Tensor
    ) -> Dict[str, Tensor]:
    """
    Forward pass retaining intermediate tensors for staged training losses.

    Expected model attributes
    -------------------------
        - masking_encoder: initial encoder block to latent masking space
        - encoder_layers: list-like container of encoder blocks
        - prototype_layers: list-like container of prototype layers
        - temporal_mixing_layers: list-like container of temporal mixing modules
        - last_layer: final projection / representation layer

    Assumptions
    -----------
        - masking_encoder, mask_token, and last_layer are accessible directly under these names 
        - encoder_layers, prototype_layers, and temporal_mixing_layers are aligned by index
        - after each encoder layer i, prototype_layers[i] is applied
        - temporal_mixing_layers[i] processes the prototype activation tensor from prototype_layers[i]
        - each temporal mixing output is pooled over time with adaptive avg pooling to produce a fixed-length vector
        - all pooled vectors are concatenated and passed to last_layer in order of prototype layer index

    Returns
    -------
    dict containing:
        - per-layer intermediates in lists:
            "z": latent outputs after each encoder block
            "a": prototype activation tensors
            "m": temporal mixing outputs
            "f": pooled feature vectors
        - "concat_features": concatenated pooled feature vector
        - "representation": final model representation
    """

    m = _unwrap_model(model)

    if not (len(m.encoder_layers) == len(m.prototype_layers) == len(m.temporal_mixing_layers)):
        raise ValueError("encoder_layers, prototype_layers, and temporal_mixing_layers must have the same length.")

    z_list = []
    a_list = []
    mix_list = []
    f_list = []
    
    # masking
    z = m.masking_encoder(x)
    if m.training: 
        mask = m.create_mask(batch_size = z.size(0), seq_len = z.size(-1), device = z.device)
        z = m.apply_latent_mask(z, mask)
    else: 
        zero_mask = torch.zeros(z.size(0), 1, z.size(-1), device = z.device, dtype = z.dtype)
        z = torch.cat([z, zero_mask], dim = 1)
    
    # encoder and prototype comparison blocks
    for encoder_layer, prototype_layer, temporal_mixing_layer in zip(m.encoder_layers, m.prototype_layers, m.temporal_mixing_layers):
        z = encoder_layer(z)
        a = prototype_layer(z)
        mix = temporal_mixing_layer(a)
        f = F.adaptive_avg_pool1d(mix, 1).squeeze(-1)

        z_list.append(z)
        a_list.append(a)
        mix_list.append(mix)
        f_list.append(f)

    # final representation
    concat_features = torch.cat(f_list, dim = 1)
    representation = m.last_layer(concat_features)

    return {
        "z": z_list,
        "a": a_list,
        "m": mix_list,
        "f": f_list,
        "concat_features": concat_features,
        "representation": representation,
    }

# ============================================================
# Cosine similarity calculation helpers
# ============================================================

def _rowise_cos_sim(
        x: Tensor, 
        y: Tensor, 
        eps: float = 1e-8
    ) -> Tensor:
    """
    Cosine similarity row-wise after flattening all non-batch dimensions.
    Returns shape: (batch,)
    """

    x = x.reshape(x.size(0), -1)
    y = y.reshape(y.size(0), -1)
    x = F.normalize(x, p = 2, dim = 1, eps = eps)
    y = F.normalize(y, p = 2, dim = 1, eps = eps)

    return torch.sum(x * y, dim = 1)

def _pairwise_cos_sim(
        x: Tensor, 
        y: Tensor, 
        eps: float = 1e-8
    ) -> Tensor:
    """
    Pairwise cosine similarity between rows of x and y.
    x: (B, D), y: (B, D) -> (B, B)
    """

    x = x.reshape(x.size(0), -1)
    y = y.reshape(y.size(0), -1)
    x = F.normalize(x, p = 2, dim = 1, eps = eps)
    y = F.normalize(y, p = 2, dim = 1, eps = eps)

    return x @ y.T

# ============================================================
# Prototype layer loss components
# ============================================================

def _prototype_contrastive_loss(
        s_pos: Tensor, 
        s_mid: Tensor, 
        s_neg: Tensor, 
        mid_weight: float = 0.5,
        neg_margin: float = 0.1
    ) -> Tensor:
    """
    Implements the prototype-space pair losses described in the draft:
        - L_pos(s) = 1 - s
        - L_mid(s) = 0.5 * (1 - s)
        - L_neg(s) = max(0, s - 0.1)

    Input shapes:
        - s_pos: (B,)
        - s_mid: (B,)
        - s_neg: (B, Nneg) or (B,)
    """

    l_pos = 1.0 - s_pos
    l_mid = mid_weight * (1.0 - s_mid)
    l_neg = torch.clamp(s_neg - neg_margin, min = 0.0)

    if l_neg.ndim == 2:
        l_neg = l_neg.mean(dim = 1)

    return l_pos + l_mid + l_neg

def _prototype_diversity_regularizer(
        prototype_vectors: Tensor, 
        threshold: float = 0.2, 
        eps: float = 1e-8
    ) -> Tensor:
    """
    Average pairwise prototype similarity penalty within a prototype bank.
        - prototype_vectors shape: (n_prototypes, channels, length)
    """

    sim = _pairwise_cos_sim(prototype_vectors, prototype_vectors, eps = eps)

    n = sim.size(0)
    if n < 2:
        return sim.new_tensor(0.0)

    off_diag_mask = ~torch.eye(n, dtype = torch.bool, device = sim.device)
    off_diag = sim[off_diag_mask]
    penalty = torch.clamp(off_diag - threshold, min=0.0)
    
    return penalty.mean()

# ============================================================
# Representation space loss
# ============================================================

def _representation_contrastive_loss(
        anc_repr: Tensor, 
        pos_repr: Tensor, 
        mid_repr: Tensor, 
        neg_repr: Tensor, 
        temperature: float = 1.0, 
        eps: float = 1e-8
    ) -> Dict[str, Tensor]:
    """
    Representation-space loss.
    
    Expected shapes
    ---------------
        - anc_repr : (B, D)
        - pos_repr : (B, D)
        - mid_repr : (B, D)
        - neg_repr : (B, K, D)
    """

    s_pos = _rowise_cos_sim(anc_repr, pos_repr, eps = eps)
    s_mid = _rowise_cos_sim(anc_repr, mid_repr, eps = eps)
    
    anc = F.normalize(anc_repr, p = 2, dim = 1, eps = eps)
    neg = F.normalize(neg_repr, p = 2, dim = 2, eps = eps)
    s_neg = torch.sum(anc.unsqueeze(1) * neg, dim = 2)

    pos_logits = s_pos / temperature
    neg_logits = s_neg / temperature

    numerator = torch.exp(pos_logits)
    denominator = numerator + torch.sum(torch.exp(neg_logits), dim = 1)

    info_nce = -torch.log(numerator / denominator).mean()
    mid_term = 0.5 * (1.0 - s_mid).mean()

    return {
        "info_nce" : info_nce,
        "mid_term" : mid_term,
        "loss" : info_nce + mid_term,
        "positive_similarity" : s_pos.mean().detach(),
        "mid_similarity" : s_mid.mean().detach(),
        "negative_similarity" : s_neg.mean().detach(),
    }

# ============================================================
# Stagewise training loss
# ============================================================

def _stage1_loss(
        model : nn.Module,
        anc_out : Dict[str, Tensor],
        pos_out : Dict[str, Tensor],
        mid_out : Dict[str, Tensor],
        neg_out : Dict[str, Tensor],
        mid_weight : float = 0.5,
        neg_margin : float = 0.1,
        diversity_threshold : float = 0.2,
        lambda_proto : float = 1.0,
        eps : float = 1e-8
    ) -> Dict[str, Tensor]:
    """
    Stage 1 prototype-space loss.

    Expected structure
    ------------------
        - anc_out["a"] : (L, ...)
        - pos_out["a"] : (L, ...)
        - mid_out["a"] : (L, ...) 
        - neg_out["a"] : (L, B, K, ...) 
        where L = number of prototype layers, B = batch size, K = number of sampled negatives per anchor, ... = prototype activation dimensions for that layer
    """

    m = _unwrap_model(model)

    if not (len(anc_out["a"]) == len(pos_out["a"]) == len(mid_out["a"]) == len(neg_out["a"]) == len(m.prototype_layers)):
        raise ValueError("Mismatch between number of prototype activation tensors and number of prototype layers.")

    layer_terms = []
    metrics = {}

    for layer_idx, proto_layer in enumerate(m.prototype_layers):
        anc_act = anc_out["a"][layer_idx]
        pos_act = pos_out["a"][layer_idx]
        mid_act = mid_out["a"][layer_idx]
        neg_act = neg_out["a"][layer_idx]

        s_pos = _rowise_cos_sim(anc_act, pos_act, eps = eps)
        s_mid = _rowise_cos_sim(anc_act, mid_act, eps = eps)

        anc_flat = anc_act.reshape(anc_act.size(0), -1)
        neg_flat = neg_act.reshape(neg_act.size(0), neg_act.size(1), -1)
        anc_flat = F.normalize(anc_flat, p = 2, dim = 1, eps = eps)
        neg_flat = F.normalize(neg_flat, p = 2, dim = 2, eps = eps)
        s_neg = torch.sum(anc_flat.unsqueeze(1) * neg_flat, dim = 2)

        pair_loss = _prototype_contrastive_loss(
            s_pos = s_pos, 
            s_mid = s_mid, 
            s_neg = s_neg, 
            mid_weight = mid_weight,
            neg_margin = neg_margin
        ).mean()
        diversity = _prototype_diversity_regularizer(
            prototype_vectors = proto_layer.prototype_vectors, 
            threshold = diversity_threshold, 
            eps = eps
        )

        total = pair_loss + lambda_proto * diversity

        layer_terms.append(total)
        metrics[f"proto_pair_loss_l{layer_idx}"] = pair_loss.detach()
        metrics[f"proto_div_l{layer_idx}"] = diversity.detach()
        metrics[f"proto_pos_sim_l{layer_idx}"] = s_pos.mean().detach()
        metrics[f"proto_mid_sim_l{layer_idx}"] = s_mid.mean().detach()
        metrics[f"proto_neg_sim_l{layer_idx}"] = s_neg.mean().detach()

    loss = torch.stack(layer_terms).sum()
    metrics["loss"] = loss

    return metrics

def _stage23_loss(
        model : nn.Module,
        anc_out : Dict[str, Tensor],
        pos_out : Dict[str, Tensor],
        mid_out : Dict[str, Tensor],
        neg_out : Dict[str, Tensor],
        temperature : float = 1.0,
        lambda_repr : float = 1.0,
        eps : float = 1e-8
    ) -> Dict[str, Tensor]:
    """
    Stage 2 / Stage 3 representation-space loss.

    Expected structure
    ------------------
        - anc_out["representation"] : (B, D)
        - pos_out["representation"] : (B, D)
        - mid_out["representation"] : (B, D)
        - neg_out["representation"] : (B, K, D)
        where B = batch size, D = representation dimension, K = number of sampled negatives per anchor
    """
    
    m = _unwrap_model(model)

    repr_terms = _representation_contrastive_loss(
        anc_repr = anc_out["representation"],
        pos_repr = pos_out["representation"],
        mid_repr = mid_out["representation"],
        neg_repr = neg_out["representation"],
        temperature = temperature,
        eps = eps
    )
    l1 = m.last_layer.weight.abs().sum()

    loss = repr_terms["loss"] + lambda_repr * l1

    return {
        "loss" : loss,
        "repr_info_nce" : repr_terms["info_nce"].detach(),
        "repr_mid_term" : repr_terms["mid_term"].detach(),
        "repr_pos_sim" : repr_terms["positive_similarity"],
        "repr_mid_sim" : repr_terms["mid_similarity"],
        "repr_neg_sim" : repr_terms["negative_similarity"],
        "last_layer_l1" : l1.detach(),
    }

# ===========================================================
# Stagewise training setup functions
# ===========================================================

def _stage_1_train(model):
    '''
    Encoder and prototype layers only
    '''

    model = _unwrap_model(model)

    model.mask_token.requires_grad = True
    for param in model.masking_encoder.parameters():
        param.requires_grad = True

    for encoder_layer in model.encoder_layers:
        for param in encoder_layer.parameters():
            param.requires_grad = True

    for prototype_layer in model.prototype_layers:
        prototype_layer.prototype_vectors.requires_grad = True

    for temporal_mixing_layer in model.temporal_mixing_layers:
        for param in temporal_mixing_layer.parameters():
            param.requires_grad = False
    
    for param in model.last_layer.parameters():
        param.requires_grad = False

def _stage_2_train(model):
    '''
    Temporal mixing and last layer only, with encoder and prototype layers frozen
    '''

    model = _unwrap_model(model)

    model.mask_token.requires_grad = False
    for param in model.masking_encoder.parameters():
        param.requires_grad = False

    for encoder_layer in model.encoder_layers:
        for param in encoder_layer.parameters():
            param.requires_grad = False

    for prototype_layer in model.prototype_layers:
        prototype_layer.prototype_vectors.requires_grad = False

    for temporal_mixing_layer in model.temporal_mixing_layers:
        for param in temporal_mixing_layer.parameters():
            param.requires_grad = True
    
    for param in model.last_layer.parameters():
        param.requires_grad = True

def _stage_3_train(model):
    '''
    Global training
    '''

    model = _unwrap_model(model)

    model.mask_token.requires_grad = True
    for param in model.masking_encoder.parameters():
        param.requires_grad = True

    for encoder_layer in model.encoder_layers:
        for param in encoder_layer.parameters():
            param.requires_grad = True

    for prototype_layer in model.prototype_layers:
        prototype_layer.prototype_vectors.requires_grad = True

    for temporal_mixing_layer in model.temporal_mixing_layers:
        for param in temporal_mixing_layer.parameters():
            param.requires_grad = True
    
    for param in model.last_layer.parameters():
        param.requires_grad = True

# ============================================================
# Optional collection hooks for future training-dynamics logging
# ============================================================

# TODO:
# Ensure memory-efficient collection (save per epoch, save detached versions of tensors)

@dataclass
class CollectorPayload:
    stage : str
    epoch : int
    batch_index : int
    is_train : bool
    anc_out : Dict[str, Tensor]
    pos_out : Dict[str, Tensor]
    mid_out : Dict[str, Tensor]
    neg_out : Dict[str, Tensor]
    loss_dict : Dict[str, Tensor]
    batch : Optional[Dict[str, Any]] = None

def _collect_training_state(
        collector_fn : Optional[Callable[[CollectorPayload], None]],
        stage : str,
        epoch : int,
        batch_index : int,
        is_train : bool,
        anc_out : Dict[str, Tensor],
        pos_out : Dict[str, Tensor],
        mid_out : Dict[str, Tensor],
        neg_out : Dict[str, Tensor],
        loss_dict : Dict[str, Tensor],
        batch : Optional[Dict[str, Any]] = None,
    ) -> None:
    """
    Hook point for future logging / checkpointing of training dynamics.

    Intended future uses include saving or summarizing:
        - prototype vectors at each epoch
        - gradients on prototype vectors
        - final representations for anchor / positive / mid / negative samples
        - prototype activations and pooled features
        - batch metadata such as sampled crop length
    """

    if collector_fn is None:
        return

    payload = CollectorPayload(
        stage = stage,
        epoch = epoch,
        batch_index = batch_index,
        is_train = is_train,
        anc_out = anc_out,
        pos_out = pos_out,
        mid_out = mid_out,
        neg_out = neg_out,
        loss_dict = loss_dict,
        batch = batch,
    )

    collector_fn(payload)

    return

# ============================================================
# Training loops 
# ============================================================

STAGE_SETTERS = {"stage1": _stage_1_train, "stage2": _stage_2_train, "stage3": _stage_3_train,}

def _run_epoch(
        model : nn.Module,
        dataloader : Iterable[Batch],
        device : torch.device,
        stage : str,
        optimizer : Optional[torch.optim.Optimizer] = None,
        mid_weight : float = 0.5,
        proto_neg_margin : float = 0.1,
        proto_diversity_threshold : float = 0.2,
        lambda_proto : float = 1.0,
        temperature : float = 1.0,
        lambda_repr : float = 1.0,
        grad_clip_norm : Optional[float] = None,
        epoch : int = 0,
        collector_fn : Optional[Callable[[CollectorPayload], None]] = None,
        use_amp : bool = True
    ) -> Dict[str, float]:
    """
    Generic epoch runner for the 3-stage training scheme with explicitly sampled
    per-anchor negatives.

    Expected batch format
    ---------------------
    batch["anchor"]   : (B, F, L)
    batch["positive"] : (B, F, L)
    batch["mid"]      : (B, F, L)
    batch["negative"] : (B, K, F, L)
    """

    # check if stage is valid and initialize training 
    if stage not in STAGE_SETTERS:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {tuple(STAGE_SETTERS.keys())}.")

    is_train = optimizer is not None
    model.train(is_train)
    STAGE_SETTERS[stage](model)

    metric_sums : Dict[str, float] = {}
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        batch = _move_to_device(batch, device)

        # extract contrastive samples from batch 
        anchor = batch["anchor"]
        positive = batch["positive"]
        mid = batch["mid"]
        negative = batch["negative"]

        if negative.ndim != 4:
            raise ValueError(f'Expected batch["negative"] to have shape (B, K, F, L), got {tuple(negative.shape)}.')

        batch_size = anchor.size(0)
        num_negatives = negative.size(1)
        negative_flat = negative.reshape(batch_size*num_negatives, negative.size(2), negative.size(3))

        if is_train:
            optimizer.zero_grad(set_to_none = True)

        autocast_enabled = bool(use_amp and device.type == "cuda")

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16, enabled = autocast_enabled):
                anc_out = _extended_forward(model, anchor)
                pos_out = _extended_forward(model, positive)
                mid_out = _extended_forward(model, mid)
                neg_out_flat = _extended_forward(model, negative_flat)

                neg_out = {}
                for key, value in neg_out_flat.items():
                    if isinstance(value, list):
                        reshaped_list = []
                        for tensor in value:
                            reshaped_tensor = tensor.reshape(batch_size, num_negatives, *tensor.shape[1:])
                            reshaped_list.append(reshaped_tensor)
                        neg_out[key] = reshaped_list
                    else:
                        neg_out[key] = value.reshape(batch_size, num_negatives, *value.shape[1:])

                if stage == "stage1":
                    loss_dict = _stage1_loss(
                        model = model,
                        anc_out = anc_out,
                        pos_out = pos_out,
                        mid_out = mid_out,
                        neg_out = neg_out,
                        mid_weight = mid_weight,
                        neg_margin = proto_neg_margin,
                        diversity_threshold = proto_diversity_threshold,
                        lambda_proto = lambda_proto
                    )
                else:
                    loss_dict = _stage23_loss(
                        model = model,
                        anc_out = anc_out,
                        pos_out = pos_out,
                        mid_out = mid_out,
                        neg_out = neg_out,
                        temperature = temperature,
                        lambda_repr = lambda_repr
                    )

                loss = loss_dict["loss"]

        if is_train:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        _collect_training_state(
            collector_fn,
            stage = stage,
            epoch = epoch,
            batch_index = batch_idx,
            is_train = is_train,
            anc_out = anc_out,
            pos_out = pos_out,
            mid_out = mid_out,
            neg_out = neg_out,
            loss_dict = loss_dict,
            batch = batch,
        )

        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach().item())

        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Dataloader produced zero batches.")

    averaged_metrics = {k : v / n_batches for k, v in metric_sums.items()}

    return averaged_metrics

def _train_wrapper(
        model : nn.Module,
        dataloader : Iterable[Batch],
        device : torch.device,
        stage : str,
        optimizer : Optional[torch.optim.Optimizer] = None,
        mid_weight : float = 0.5,
        proto_neg_margin : float = 0.1,
        proto_diversity_threshold : float = 0.2,
        lambda_proto : float = 1.0,
        temperature : float = 1.0,
        lambda_repr : float = 1.0,
        grad_clip_norm : Optional[float] = None,
        epoch : int = 0,
        collector_fn : Optional[Callable[[CollectorPayload], None]] = None,
        use_amp : bool = True
    ) -> Dict[str, float]:
    
    model.train()
    return _run_epoch(
        model = model,
        dataloader = dataloader,
        device = device,
        stage = stage,
        optimizer = optimizer,
        mid_weight = mid_weight,
        proto_neg_margin = proto_neg_margin,
        proto_diversity_threshold = proto_diversity_threshold,
        lambda_proto = lambda_proto,
        temperature = temperature,
        lambda_repr = lambda_repr,
        grad_clip_norm = grad_clip_norm,
        epoch = epoch,
        collector_fn = collector_fn,
        use_amp = use_amp
    )

@torch.no_grad()
def _evaluate_wrapper(
        model: nn.Module,
        dataloader: Iterable[Batch],
        device: torch.device,
        stage: str,
        *,
        mid_weight : float = 0.5,
        proto_neg_margin: float = 0.1,
        proto_diversity_threshold: float = 0.2,
        lambda_proto: float = 1.0,
        temperature: float = 1.0,
        lambda_repr: float = 1.0,
        epoch: int = 0,
        collector_fn: Optional[Callable[[CollectorPayload], None]] = None,
        use_amp: bool = True
    ) -> Dict[str, float]:

    model.eval()
    return _run_epoch(
        model = model,
        dataloader = dataloader,
        device = device,
        stage = stage,
        optimizer = None,
        mid_weight = mid_weight,
        proto_neg_margin = proto_neg_margin,
        proto_diversity_threshold = proto_diversity_threshold,
        lambda_proto = lambda_proto,
        temperature = temperature,
        lambda_repr = lambda_repr,
        grad_clip_norm = None,
        epoch = epoch,
        collector_fn = collector_fn,
        use_amp = use_amp
    )

# ============================================================
# Training orchestration
# ============================================================

def save_checkpoint(model: nn.Module, path: str) -> None:
    state_dict = _unwrap_model(model).state_dict()
    torch.save(state_dict, path)

def run_training(
        model: nn.Module,
        train_loader: Iterable[Batch],
        val_loader: Optional[Iterable[Batch]],
        device: torch.device,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        epochs_stage1: int,
        epochs_stage2: int,
        epochs_stage3: int,
        scheduler_dict: Optional[Dict[str, Any]] = None,
        mid_weight: float = 0.5,
        proto_neg_margin: float = 0.1,
        proto_diversity_threshold: float = 0.2,
        lambda_proto: float = 1.0,
        temperature: float = 1.0,
        lambda_repr: float = 1.0,
        grad_clip_norm: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_epochs: Optional[List[int]] = None,
        collector_fn: Optional[Callable[[CollectorPayload], None]] = None,
        use_amp : bool = True
    ) -> List[Dict[str, Any]]:
    """
    Full three-stage training loop.

    optimizer_dict must contain: "stage1", "stage2", "stage3"

    scheduler_dict is optional and may contain matching keys
    """

    required_optimizers = {"stage1", "stage2", "stage3"}
    missing = required_optimizers - set(optimizer_dict.keys())
    if missing:
        raise KeyError(f"optimizer_dict is missing required keys: {missing}")

    scheduler_dict = scheduler_dict or {}
    history: List[Dict[str, Any]] = []

    stage_plan = [("stage1", epochs_stage1), ("stage2", epochs_stage2), ("stage3", epochs_stage3)]
    global_epoch = 0
    total_epochs = epochs_stage1 + epochs_stage2 + epochs_stage3
    print(f"Training for {total_epochs} epochs.")

    for stage_name, n_epochs in stage_plan:
        if n_epochs <= 0:
            continue

        print(f"\n=== {stage_name} ({n_epochs} epochs) ===")

        for local_epoch in range(n_epochs):
            print(f"[{stage_name}] epoch {local_epoch + 1}/{n_epochs} | global {global_epoch + 1}/{total_epochs}")

            train_metrics = _train_wrapper(
                model = model,
                dataloader = train_loader,
                device = device,
                stage = stage_name,
                optimizer = optimizer_dict[stage_name],
                mid_weight = mid_weight,
                proto_neg_margin = proto_neg_margin,
                proto_diversity_threshold = proto_diversity_threshold,
                lambda_proto = lambda_proto,
                temperature = temperature,
                lambda_repr = lambda_repr,
                grad_clip_norm = grad_clip_norm,
                epoch = global_epoch,
                collector_fn = collector_fn,
                use_amp = use_amp
            )

            val_metrics = None
            if val_loader is not None:
                val_metrics = _evaluate_wrapper(
                    model = model,
                    dataloader = val_loader,
                    device = device,
                    stage = stage_name,
                    mid_weight = 0.5,
                    proto_neg_margin = proto_neg_margin,
                    proto_diversity_threshold = proto_diversity_threshold,
                    lambda_proto = lambda_proto,
                    temperature = temperature,
                    lambda_repr = lambda_repr,
                    epoch = global_epoch,
                    collector_fn = collector_fn,
                    use_amp = use_amp
                )

            scheduler = scheduler_dict.get(stage_name, None)
            if scheduler is not None:
                scheduler.step()

            record = {
                "global_epoch": global_epoch,
                "stage_epoch": local_epoch,
                "stage": stage_name,
                "train": train_metrics,
                "val": val_metrics
            }
            history.append(record)

            train_loss = train_metrics.get("loss", float("nan"))
            msg = f"  train loss: {train_loss:.6f}"
            if val_metrics is not None:
                msg += f" | val loss: {val_metrics.get('loss', float('nan')):.6f}"
            print(msg)

            if checkpoint_epochs is not None and global_epoch in set(checkpoint_epochs):
                checkpoint_file = f"{checkpoint_path}_epoch{global_epoch}.pt" if checkpoint_path else None
                if checkpoint_file:
                    save_checkpoint(model, checkpoint_file)
                    print(f"Saved checkpoint at epoch {global_epoch} to {checkpoint_file}")

            global_epoch += 1

    if checkpoint_path is not None:
        save_checkpoint(model, checkpoint_path)

    return history