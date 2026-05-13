"""Runtime monkey-patches for upstream Composer bugs we hit in multi-node training.

These patches are intentionally narrow and conservative; each one documents the
upstream bug it works around and is safe to drop once Composer ships a fix.

Patches applied
---------------

1. ``composer.utils.dist.get_node_signal_file_name``
   Upstream computes a per-rank filename of the form
   ``._signal_file_node{node_rank}_{random_string}`` and then does
   ``broadcast_object_list(file_name_list, src=0)`` to make every rank agree.
   The broadcast overwrites *the whole filename* with global-rank-0's value,
   so every node ends up with the SAME signal-file path -- including the
   ``node{node_rank}`` part, which is misleadingly always ``node0``.

   When ``save_folder`` lives on a filesystem that is shared across nodes
   (NFS, Lustre, ...), every node's local-rank-0 writes/deletes the same
   file. The write is idempotent ("last writer wins"), but the
   ``os.remove(signal_file_path)`` in
   ``composer.trainer.trainer.Trainer._get_autoresume_checkpoint`` (and the
   parallel code path in ``composer/utils/checkpoint.py``) is racy: the
   first node deletes the file, the others raise ``FileNotFoundError`` and
   crash, which in turn hangs the rest of the world on the next
   ``dist.barrier()`` until the NCCL watchdog times out (~5 min).

   The patched version broadcasts only the random suffix so all ranks on
   all nodes still agree on a single per-job suffix, but composes the
   final filename locally with the *local* ``node_rank`` -- so each node
   really does get its own filename, and each local-rank-0's
   ``os.remove`` only touches that node's own file. Single-node behaviour
   is unchanged (node_rank is always 0).
"""

from __future__ import annotations

import random
import string
import logging

log = logging.getLogger(__name__)

_PATCHED: bool = False


def _patched_get_node_signal_file_name(rng=None):
    from composer.utils import dist as composer_dist

    if rng is None:
        rng = random.Random()

    random_string = ''.join(rng.choices(string.ascii_letters + string.digits, k=6))
    node_rank = composer_dist.get_node_rank()

    suffix_list = [random_string]
    composer_dist.broadcast_object_list(suffix_list, src=0)
    shared_suffix = suffix_list[0]

    return f'._signal_file_node{node_rank}_{shared_suffix}'


def apply_composer_patches() -> None:
    """Apply all Composer monkey-patches. Idempotent; safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return

    from composer.utils import dist as composer_dist

    composer_dist.get_node_signal_file_name = _patched_get_node_signal_file_name

    _PATCHED = True
    log.info(
        "Applied Composer monkey-patches: get_node_signal_file_name now returns "
        "per-node unique filenames (multi-node autoresume / NFS signal-file race fix)."
    )
