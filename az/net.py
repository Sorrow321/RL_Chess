"""Policy+value network (docs/03-net.md).

AlphaZero-style ResNet, deliberately small: 6 residual blocks of 128 channels,
~1.9M parameters. The inference cost of this net is the number that rules the
whole system — one forward per runner cycle serves every game in the fleet, so
every millisecond here multiplies by the self-play budget (bench/net_bench.py
measures it; docs/03 has the targets).

    input  (B, 19, 8, 8) uint8 planes, oriented to the side to move
    stem   conv3x3 19->128, BN, ReLU
    tower  6 x [conv3x3, BN, ReLU, conv3x3, BN, +skip, ReLU]
    policy conv1x1 128->73 -> (B, 4672) logits
    value  conv1x1 128->8, BN, ReLU, flatten(512) -> fc 256 -> ReLU -> fc 1 -> tanh

The policy head is spatial on purpose. The move encoding of docs/01 is
`index = from_square * 73 + move_type`, i.e. it is already a 73-plane stack
over the 8x8 board, so a conv1x1 producing exactly that stack scores each
piece's moves from the cell the piece stands on. ~9k parameters instead of the
~2.4M an FC head would need, and no capacity spent learning that from-squares
matter. `spatial_to_policy` is the one place that flattening happens and the
one thing that has to agree with the C++ encoder — tests/net_test.py checks it
against `chess_engine.move_to_index` rather than against itself.

Losses (docs/03): `L = policy_loss + lambda * F.mse_loss(v, z)`, lambda=1 to
start. Illegal moves are masked out of the softmax at *training* time too, so
the net never spends capacity suppressing moves the search can never take.
"""
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mirrors of the C++ encoder's constants (chess_engine.NUM_PLANES /
# .POLICY_SIZE). Duplicated rather than imported so this module stays pure
# torch; tests/net_test.py asserts the two agree.
NUM_PLANES = 19
MOVE_TYPES = 73
POLICY_SIZE = 64 * MOVE_TYPES  # 4672

# Plane 18 holds min(halfmove_clock, 100) as a constant plane; everything else
# the encoder emits is already 0/1.
HALFMOVE_PLANE = 18
HALFMOVE_MAX = 100.0

VALUE_CHANNELS = 8    # conv1x1 128->8 -> flatten(512)
VALUE_HIDDEN = 256


class ResidualBlock(nn.Module):
    """conv-BN-ReLU-conv-BN, skip, ReLU. Convs are bias-free (BN follows)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(x + y, inplace=True)


def spatial_to_policy(head_map):
    """(B, 73, 8, 8) policy planes -> (B, 4672) logits.

    Index layout must be `from_square * 73 + move_type` with
    `from_square = rank * 8 + file` in the mover's orientation — the exact
    convention of `az::move_to_index` (cpp/encoding.hpp). The transpose is the
    whole conversion: plane-major (type, square) becomes square-major
    (square, type).
    """
    b = head_map.shape[0]
    return head_map.reshape(b, MOVE_TYPES, 64).transpose(1, 2).reshape(b, POLICY_SIZE)


class PolicyValueNet(nn.Module):
    """The v1 net. Growth path is one axis at a time: 6x128 -> 8x160 -> 10x192."""

    def __init__(self, channels=128, blocks=6):
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # No BN on the policy conv: it *is* the logit layer, and normalizing
        # logits would fight the softmax temperature the search relies on.
        self.policy_conv = nn.Conv2d(channels, MOVE_TYPES, 1)

        self.value_conv = nn.Conv2d(channels, VALUE_CHANNELS, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(VALUE_CHANNELS)
        self.value_fc1 = nn.Linear(VALUE_CHANNELS * 64, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x):
        """x: (B, 19, 8, 8) float. Returns (logits (B, 4672), value (B,))."""
        h = self.tower(self.stem(x))

        logits = spatial_to_policy(self.policy_conv(h))

        v = F.relu(self.value_bn(self.value_conv(h)), inplace=True).flatten(1)
        v = F.relu(self.value_fc1(v), inplace=True)
        return logits, torch.tanh(self.value_fc2(v)).squeeze(-1)


def param_count(net):
    return sum(p.numel() for p in net.parameters())


# --- input ------------------------------------------------------------------

def planes_to_tensor(planes, device, dtype=torch.float32, channels_last=True):
    """(B, 19, 8, 8) uint8 planes -> float tensor on `device`.

    Takes the uint8 array straight from `Runner.pending()` / `encode_batch()`
    and moves it as uint8 (a quarter of the PCIe traffic of float32), widening
    on the far side. The halfmove plane is the only non-binary one; it is
    divided by 100 so every input channel lives in [0, 1].
    """
    x = torch.from_numpy(planes) if isinstance(planes, np.ndarray) else planes
    if x.dtype != torch.uint8:
        raise TypeError(f"expected uint8 planes from the encoder, got {x.dtype}")
    x = x.to(device, non_blocking=True).to(dtype)
    # Safe in-place: the dtype conversion above always produced a fresh tensor.
    x[:, HALFMOVE_PLANE] *= 1.0 / HALFMOVE_MAX
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


# --- losses -----------------------------------------------------------------

def policy_loss(logits, legal_idx, target, valid=None):
    """Cross-entropy over a *padded ragged* legal-move set — the training path.

    The runner dumps `legal_idx`/`visits` per move (docs/02), so batches are
    ragged: pad to a common width L and pass `valid` to mark the real entries.
    Padded slots are excluded from the softmax exactly as illegal moves are.

    logits    (B, 4672) raw head output
    legal_idx (B, L) int64 indices into the policy head
    target    (B, L) float, each row summing to 1 (visits/sum, or one-hot for
              a hard bootstrap label)
    valid     (B, L) bool, or None when every row is full width
    """
    gathered = logits.gather(1, legal_idx)
    if valid is not None:
        gathered = gathered.masked_fill(~valid, float("-inf"))
    logp = F.log_softmax(gathered, dim=1)
    if valid is not None:
        # -inf * 0 is NaN, so zero the pad slots before weighting by target.
        logp = logp.masked_fill(~valid, 0.0)
    return -(target * logp).sum(1).mean()


def dense_policy_loss(logits, target, legal_mask):
    """Same loss over dense (B, 4672) targets — for tooling and tests.

    Equivalent to `policy_loss` and far more memory-hungry; the ragged form is
    what the training loop should use.
    """
    logp = F.log_softmax(logits.masked_fill(~legal_mask, float("-inf")), dim=1)
    return -(target * logp.masked_fill(~legal_mask, 0.0)).sum(1).mean()


# --- inference --------------------------------------------------------------

class Evaluator:
    """The net as the runner sees it: uint8 planes in, (priors, values) out.

    This is the seam docs/02's loop plugs into:

        ev = Evaluator(net)
        while (batch := runner.pending()).size:
            runner.feed(*ev(batch))

    Priors come back as a full 4672-wide softmax in float32 — the runner masks
    to legal moves and renormalizes in C++, so nothing here needs to know what
    is legal. fp16 weights and channels_last are the self-play defaults
    (docs/03); the softmax and tanh are taken in float32 regardless, which
    costs nothing at B x 4672 and keeps half-precision out of the numerics the
    search actually consumes.

    It holds its own half-precision copy of the weights: `nn.Module.to(dtype)`
    is in-place, and silently truncating the trainer's fp32 master weights the
    moment self-play starts is not a bug worth owning. Call `refresh(net)` to
    re-sync after a generation of training.
    """

    def __init__(self, net, device=None, dtype=torch.float16, channels_last=True, clone=True):
        if device is None:
            device = next(net.parameters()).device
        self.device = torch.device(device)
        if self.device.type != "cuda" and dtype is torch.float16:
            dtype = torch.float32  # fp16 on CPU is emulated and slower than fp32
        self.dtype = dtype
        self.channels_last = channels_last

        net = deepcopy(net) if clone else net
        net = net.to(device=self.device, dtype=self.dtype)
        if channels_last:
            net = net.to(memory_format=torch.channels_last)
        self.net = net.eval()

    def refresh(self, net):
        """Copy weights in from the fp32 training net (casts dtype and device)."""
        self.net.load_state_dict(net.state_dict())
        self.net.eval()
        return self

    @classmethod
    def from_checkpoint(cls, path, device=None, dtype=torch.float16, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs.setdefault("clone", False)  # the net was just built here; nobody else holds it
        return cls(load_checkpoint(path, device), device=device, dtype=dtype, **kwargs)

    @torch.inference_mode()
    def __call__(self, planes):
        """(B, 19, 8, 8) uint8 -> (priors (B, 4672) float32, values (B,) float32)."""
        n = planes.shape[0]
        if n == 0:
            return (np.empty((0, POLICY_SIZE), dtype=np.float32),
                    np.empty(0, dtype=np.float32))
        x = planes_to_tensor(planes, self.device, self.dtype, self.channels_last)
        logits, v = self.net(x)
        p = torch.softmax(logits.float(), dim=1)
        return (np.ascontiguousarray(p.cpu().numpy()),
                np.ascontiguousarray(v.float().clamp_(-1.0, 1.0).cpu().numpy()))


# --- checkpoints ------------------------------------------------------------
# RL_2048's convention: {"state_dict": ...} in gen_NNN.pt / latest.pt, so the
# run-management tooling ports over unchanged. The architecture fields ride
# along so a checkpoint from a widened net (docs/05 plateau playbook step 4)
# still loads without being told its shape.

def save_checkpoint(net, path, **extra):
    torch.save({"state_dict": net.state_dict(), "channels": net.channels,
                "blocks": net.blocks, **extra}, path)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint into a net in eval mode (call .train() to fine-tune)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = PolicyValueNet(channels=ckpt.get("channels", 128),
                         blocks=ckpt.get("blocks", 6)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval()
