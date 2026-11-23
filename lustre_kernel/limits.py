"""
lustre_kernel.limits

The hardware ceilings that both the kernel and the indexer must
respect. The dependency runs indexer -> lustre_kernel, so the indexer
can import these without a cycle.

Why a constant and not a comment: the batch ceiling has to hold on
EVERY path to the GPU, and the paths are separate files written at
different times. A ceiling enforced at one call site is a ceiling on
that call site only.
"""

from __future__ import annotations

__all__ = ["MAX_EMBED_BATCH_SIZE"]


# The largest batch of images that may reach the GPU in one forward
# pass.
#
# The activation memory of the ViT scales with the batch, and the
# card is a 10 GiB RTX 3080. At batch 16 the 640-input run left
# ~90 MiB free while another process held the GPU, and that is the
# condition that took the card off the PCIe bus (Xid 79, "GPU has
# fallen off the bus") — a fault that only a NODE REBOOT recovers.
#
# A smaller batch costs throughput, and throughput is cheap to give
# up. A crashed card is not.
#
# Measured on this card (RTX 3080, 10240 MiB) with C-RADIOv4,
# worst case (every photo the same shape, so the whole batch is one
# forward pass):
#
#   model input  batch   peak VRAM
#   480            8      4856 MB
#   640            4      4828 MB
#   640            8      5106 MB
#   640           16      5671 MB   <- never run this
#
# The indexer's CLI/env clamp is in indexer/local_sync.py; the
# embedder's own is in lustre_kernel/_radio_models.py. Both read this
# constant, so they cannot drift apart.
MAX_EMBED_BATCH_SIZE = 8
