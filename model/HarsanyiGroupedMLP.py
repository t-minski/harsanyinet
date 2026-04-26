import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

"""
Grouped Harsanyi-MLP for MARL-style inputs.

Key difference from tabular Harsanyi-MLP:
- Players are agents (N), not scalar features (N*d).
- First-layer tau selects agent subsets per unit: tau shape [hidden_dim, N].
- AND gate checks agent activation via ||x_i||_1 over per-agent feature vectors.
"""

EPS = 1e-30


def init_layer(layer: nn.Module, weight_init: str) -> None:
    if not isinstance(layer, nn.Linear):
        return
    if weight_init == "xavier":
        nn.init.xavier_uniform_(layer.weight)
    elif weight_init == "uniform":
        nn.init.uniform_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: Tensor, beta: int = 1, slope: int = 1) -> Tensor:
        ctx.save_for_backward(input_)
        ctx.slope = slope
        ctx.beta = beta
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            ctx.beta
            * grad_input
            * ctx.slope
            * torch.exp(-ctx.slope * input_)
            / ((torch.exp(-ctx.slope * input_) + 1) ** 2 + EPS)
        )
        return grad, None


class StraightThroughEstimator(nn.Module):
    def forward(self, x: Tensor, beta: int = 1) -> Tensor:
        return STEFunction.apply(x, beta)


class GroupedInputHarsanyiBlock(nn.Module):
    """
    First Harsanyi block for grouped inputs.

    Input:
      x: [batch, n_players, player_dim]
    Internal:
      tau mask over players: [output_dim, n_players]
      scalar-expanded mask for linear op: [output_dim, n_players * player_dim]
    Output:
      y: [batch, output_dim], delta: [batch, output_dim]
    """

    def __init__(
        self,
        n_players: int,
        player_dim: int,
        output_dim: int,
        device: str = "cuda:0",
        beta: int = 10,
        gamma: int = 100,
        initial_V: float = 1.0,
        act_ratio: float = 0.1,
        weight_init: str = "xavier",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.n_players = n_players
        self.player_dim = player_dim
        self.flat_input_dim = n_players * player_dim
        self.output_dim = output_dim
        self.device = device
        self.initial_V = initial_V
        self.act_ratio = act_ratio

        # tau over players, not over scalar dimensions.
        self.v = nn.Linear(n_players, output_dim, bias=False)
        self.fc = nn.Linear(self.flat_input_dim, output_dim, bias=False)
        self.ste = StraightThroughEstimator()
        self.activation = nn.functional.relu

        self.init_v()
        init_layer(self.fc, weight_init)

    def init_v(self) -> None:
        self.v.weight.data[:, :] = -self.initial_V
        active = max(1, int(self.act_ratio * self.n_players))
        for i in range(self.output_dim):
            act_index = np.random.choice(self.n_players, active, replace=False)
            self.v.weight.data[i][act_index] = self.initial_V

    def _expand_player_mask(self, player_mask: Tensor) -> Tensor:
        # [output_dim, n_players] -> [output_dim, n_players, player_dim]
        return player_mask.unsqueeze(-1).expand(-1, -1, self.player_dim)

    def _layer(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        v_is_child = self.ste(self.v.weight, beta=self.beta)  # [O, N]
        v_is_not_child = torch.ones_like(v_is_child) - v_is_child

        # Expand player-level mask to scalar feature mask for linear op.
        scalar_mask = self._expand_player_mask(v_is_child).reshape(
            self.output_dim, self.flat_input_dim
        )
        x_flat = x.reshape(batch_size, self.flat_input_dim)
        weight = self.fc.weight * scalar_mask
        y = torch.matmul(x_flat, weight.transpose(0, 1))

        # Agent-level AND gate: active if L1 norm over player features is non-zero.
        x_enlarge = x.unsqueeze(1).expand(-1, self.output_dim, -1, -1)
        player_activation = torch.norm(x_enlarge, p=1, dim=-1)  # [B, O, N]
        delta_en = torch.tanh(self.gamma * torch.abs(player_activation))
        delta_en = (
            delta_en * v_is_child.unsqueeze(0) + v_is_not_child.unsqueeze(0)
        )
        delta = torch.prod(delta_en, dim=-1)  # [B, O]
        y = y * delta
        return y, delta

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y, delta = self._layer(x)
        if self.activation is not None:
            y = self.activation(y)
        return y, delta


class HiddenHarsanyiBlock(nn.Module):
    """
    Standard Harsanyi hidden block after grouped first layer.

    Operates on hidden vectors [batch, hidden_dim] exactly like the tabular variant.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: str = "cuda:0",
        beta: int = 10,
        gamma: int = 100,
        initial_V: float = 1.0,
        act_ratio: float = 0.1,
        weight_init: str = "xavier",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.initial_V = initial_V
        self.act_ratio = act_ratio

        self.v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.ste = StraightThroughEstimator()
        self.activation = nn.functional.relu

        self.init_v()
        init_layer(self.fc, weight_init)

    def init_v(self) -> None:
        self.v.weight.data[:, :] = -self.initial_V
        active = max(1, int(self.act_ratio * self.input_dim))
        for i in range(self.output_dim):
            act_index = np.random.choice(self.input_dim, active, replace=False)
            self.v.weight.data[i][act_index] = self.initial_V

    def _extend_layer(self, x: Tensor) -> Tensor:
        return x.unsqueeze(1).repeat(1, self.output_dim, 1)

    def _get_trigger_value(self, input_: Tensor) -> Tensor:
        v = self.v.weight
        v_is_child = self.ste(v, beta=self.beta)
        v_is_not_child = torch.ones_like(v_is_child) - v_is_child
        delta_en = torch.tanh(self.gamma * torch.abs(input_))
        delta_en = delta_en * v_is_child + v_is_not_child
        return torch.prod(delta_en, dim=-1)

    def _layer(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_enlarge = self._extend_layer(x)
        x_enlarge = x_enlarge * self.ste(self.v.weight, beta=self.beta)
        weight = self.fc.weight * self.ste(self.v.weight, beta=self.beta)
        y = torch.matmul(x, weight.transpose(0, 1))
        delta = self._get_trigger_value(x_enlarge)
        y = y * delta
        return y, delta

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y, delta = self._layer(x)
        if self.activation is not None:
            y = self.activation(y)
        return y, delta


class HarsanyiGroupedNet(nn.Module):
    """
    Grouped Harsanyi network with per-agent Shapley decomposition support.

    The first block preserves player semantics (agents), while deeper blocks
    compose interactions over hidden units.
    """
    def __init__(
        self,
        n_players: int,
        player_dim: int,
        num_classes: int = 1,
        num_layers: int = 1,
        hidden_dim: int = 100,
        beta: int = 10,
        gamma: int = 100,
        initial_V: float = 1.0,
        act_ratio: float = 0.1,
        device: str = "cuda:0",
        weight_init: str = "xavier",
    ) -> None:
        super().__init__()
        assert num_layers > 0, "should have at least one Harsanyi layer"

        self.n_players = n_players
        self.player_dim = player_dim
        self.input_dim = n_players * player_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.HarsanyiBlocks = nn.ModuleList()
        self.fc = nn.ModuleList()

        first_layer_act_ratio = max((1 + EPS) / self.n_players, act_ratio)
        first_block = GroupedInputHarsanyiBlock(
            n_players=n_players,
            player_dim=player_dim,
            output_dim=hidden_dim,
            device=device,
            beta=beta,
            gamma=gamma,
            initial_V=initial_V,
            act_ratio=first_layer_act_ratio,
            weight_init=weight_init,
        )
        self.HarsanyiBlocks.append(first_block)
        self.fc.append(nn.Linear(hidden_dim, num_classes, bias=False))

        for _ in range(1, num_layers):
            block = HiddenHarsanyiBlock(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                device=device,
                beta=beta,
                gamma=gamma,
                initial_V=initial_V,
                act_ratio=act_ratio,
                weight_init=weight_init,
            )
            self.HarsanyiBlocks.append(block)
            self.fc.append(nn.Linear(hidden_dim, num_classes, bias=False))

        self.init_weights()

    def init_weights(self) -> None:
        for layer in self.fc:
            init_layer(layer, "xavier")

    def forward(self, x: Tensor) -> Tensor:
        hidden_y = None
        z = x
        for layer_index in range(self.num_layers):
            layer = self.HarsanyiBlocks[layer_index]
            z, _ = layer(z)
            y_ = self.fc[layer_index](z)
            if hidden_y is None:
                hidden_y = y_
            else:
                hidden_y += y_
        return hidden_y

    def _get_value(
        self, x: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor]]:
        """
        Forward pass with decomposition traces.

        Returns:
          output: [batch, 1]
          zs: post-ReLU unit activations per layer, each [batch, hidden_dim]
          ys: per-layer readout contribution before summation, each [batch, 1]
          deltas: AND gate activations per layer, each [batch, hidden_dim]
        """
        deltas: list[Tensor] = []
        zs: list[Tensor] = []
        ys: list[Tensor] = []
        hidden_y = None
        z = x
        for layer_index in range(self.num_layers):
            layer = self.HarsanyiBlocks[layer_index]
            z, delta = layer(z)
            y_ = self.fc[layer_index](z)
            zs.append(z)
            ys.append(y_)
            deltas.append(delta)
            if hidden_y is None:
                hidden_y = y_
            else:
                hidden_y += y_
        return hidden_y, zs, ys, deltas

    def get_tau_masks(self) -> list[torch.Tensor]:
        """Return frozen binary tau masks, one [out_dim, in_dim] tensor per layer."""
        tau_masks: list[torch.Tensor] = []
        for block in self.HarsanyiBlocks:
            tau_masks.append((block.v.weight.detach() > 0).to(torch.bool))
        return tau_masks

    def get_receptive_fields(self) -> list[list[set[int]]]:
        """
        Build receptive fields over agent players from frozen tau masks.

        Layer 0:
          receptive field of a unit is the set of players with tau > 0.
        Deeper layers:
          receptive field is union of receptive fields of selected children units.

        Returns:
          receptive_fields[layer][unit] -> set[player_index]
        """
        tau_masks = self.get_tau_masks()
        receptive_fields: list[list[set[int]]] = []

        for layer_index, tau in enumerate(tau_masks):
            layer_fields: list[set[int]] = []
            if layer_index == 0:
                for unit_mask in tau:
                    rf = {idx for idx, flag in enumerate(unit_mask.tolist()) if flag}
                    layer_fields.append(rf)
            else:
                prev_fields = receptive_fields[layer_index - 1]
                for unit_mask in tau:
                    child_indices = [idx for idx, flag in enumerate(unit_mask.tolist()) if flag]
                    rf: set[int] = set()
                    for child_idx in child_indices:
                        rf.update(prev_fields[child_idx])
                    layer_fields.append(rf)
            receptive_fields.append(layer_fields)
        return receptive_fields

    def shapley_values(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute per-agent Shapley values via Theorem 3 decomposition.

        For each unit contribution w_u * z_u, distribute equally across players
        in its receptive field R_u: (w_u * z_u) / |R_u|.

        Returns:
          phi: [batch, n_players]
          output: [batch, 1]
        """
        output, zs, _, _ = self._get_value(x)
        receptive_fields = self.get_receptive_fields()
        phi = torch.zeros((x.shape[0], self.n_players), device=x.device, dtype=output.dtype)

        for layer_index, z in enumerate(zs):
            # [num_classes, hidden_dim] -> assume scalar output (num_classes=1)
            w = self.fc[layer_index].weight[0]  # [hidden_dim]
            unit_contrib = z * w.unsqueeze(0)  # [batch, hidden_dim]
            for unit_index, rf in enumerate(receptive_fields[layer_index]):
                if not rf:
                    continue
                share = unit_contrib[:, unit_index] / float(len(rf))
                for player_idx in rf:
                    phi[:, player_idx] += share
        return phi, output
