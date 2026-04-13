import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    1D Residual Block

    Parameters
    ----------
    layer_configs : list[dict]
        One dict per Conv1d layer. Each dict is passed to nn.Conv1d.
        Required keys per dict:
            - in_channels
            - out_channels
            - kernel_size
        Optional keys:
            - stride
            - padding
            - dilation
            - groups
            - bias
            - padding_mode

    activation_type : str, default = "post"
        Either:
            - "post" : ResNet v1 style (x -> conv -> norm -> act -> conv -> norm -> +id -> act)
            - "pre"  : ResNet v2 style (x -> norm -> act -> conv -> norm -> act -> conv -> +id)

    norm_layer : callable, default = nn.BatchNorm1d
        Normalization constructor. Set to None to disable normalization.

    activation : callable, default = nn.ReLU
        Activation constructor.

    downsample_config : dict | None, default = None
        Optional explicit config for the skip projection Conv1d.
        If omitted, a 1x1 projection is created automatically when needed.

    final_activation : bool | None, default = None
        Whether to apply activation after residual addition.
        Defaults:
            - True for post
            - False for pre
    """

    def __init__(
        self,
        layer_configs,
        activation_type = "pre",
        norm_layer = nn.BatchNorm1d,
        activation = nn.ReLU,
        downsample_config = None,
        final_activation = None,
    ):
        
        super().__init__()

        if not layer_configs:
            raise ValueError("layer_configs must contain at least one layer config.")
        if activation_type not in {"post", "pre"}:
            raise ValueError("activation_type must be either 'post' or 'pre'.")

        self.activation_type = activation_type
        self.norm_layer = norm_layer
        self.activation_cls = activation
        self.activation = activation()
        if final_activation is None:
            self.final_activation = (activation_type == "post")
        else:
            self.final_activation = final_activation

        self.layers = nn.ModuleList()

        for i, cfg in enumerate(layer_configs):
            
            required = {"in_channels", "out_channels", "kernel_size"}
            missing = required - set(cfg)
            if missing:
                raise ValueError(f"Layer config at index {i} is missing required keys: {missing}")

            cfg = cfg.copy()
            in_ch = cfg["in_channels"]
            out_ch = cfg["out_channels"]

            if "padding" not in cfg:
                k = cfg["kernel_size"]
                d = cfg.get("dilation", 1)

                if isinstance(k, tuple):
                    raise ValueError("This ResidualBlock expects Conv1d kernel_size to be an int.")
                cfg["padding"] = d * (k - 1) // 2

            conv = nn.Conv1d(**cfg)

            if activation_type == "post":
                parts = [conv]
                if norm_layer is not None:
                    parts.append(norm_layer(out_ch))
                self.layers.append(nn.Sequential(*parts))

            else:
                parts = []
                if norm_layer is not None:
                    parts.append(norm_layer(in_ch))
                parts.append(activation())
                parts.append(conv)
                self.layers.append(nn.Sequential(*parts))

        first_cfg = layer_configs[0]
        last_cfg = layer_configs[-1]
        input_channels = first_cfg["in_channels"]
        output_channels = last_cfg["out_channels"]

        total_stride = 1
        for cfg in layer_configs:
            total_stride *= cfg.get("stride", 1)

        self.down = None

        if input_channels != output_channels or total_stride != 1:
            if downsample_config is None:
                self.down = nn.Sequential(
                    nn.Conv1d(
                        in_channels = input_channels,
                        out_channels = output_channels,
                        kernel_size = 1,
                        stride = total_stride,
                        bias = False,
                    ),
                    norm_layer(output_channels) if norm_layer is not None else nn.Identity(),
                )
            
            else:
                ds_cfg = downsample_config.copy()
                ds_cfg.setdefault("in_channels", input_channels)
                ds_cfg.setdefault("out_channels", output_channels)
                ds_cfg.setdefault("kernel_size", 1)
                ds_cfg.setdefault("stride", total_stride)

                self.down = nn.Sequential(
                    nn.Conv1d(**ds_cfg),
                    norm_layer(ds_cfg["out_channels"]) if norm_layer is not None else nn.Identity(),
                )

    def forward(self, x):
        
        identity = x if self.down is None else self.down(x)

        out = x

        if self.activation_type == "post":
            # ResNet v1: [conv -> norm -> relu] ... last layer has no relu before addition
            for i, layer in enumerate(self.layers):
                out = layer(out)
                if i < len(self.layers) - 1:
                    out = self.activation(out)

        else:
            # ResNet v2: [norm -> relu -> conv] for every layer
            for layer in self.layers:
                out = layer(out)

        out = out + identity

        if self.final_activation:
            out = self.activation(out)

        return out