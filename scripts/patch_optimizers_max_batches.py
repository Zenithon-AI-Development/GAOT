"""Apply max_train_batches logic to optimizers.py in-place when run from repo root. Idempotent."""
import pathlib

path = pathlib.Path("src/utils/optimizers.py")
s = path.read_text()
lines = s.splitlines(keepends=True)

idx_for = next(
    (i for i, l in enumerate(lines) if "for batch in trainer.train_loader:" in l and not l.lstrip().startswith("#")),
    None,
)
if idx_for is None or idx_for + 1 >= len(lines):
    exit(1)

# Revert any existing patch block so we can apply cleanly (removes max_batches, num_batches_done, if/break).
if idx_for >= 2 and idx_for + 2 < len(lines):
    if "max_batches = getattr" in lines[idx_for - 2] and "num_batches_done = 0" in lines[idx_for - 1]:
        if "if max_batches is not None" in lines[idx_for + 1] and "break" in lines[idx_for + 2]:
            lines = lines[: idx_for - 2] + [lines[idx_for], lines[idx_for + 3]] + lines[idx_for + 4 :]
            s = "".join(lines)
            lines = s.splitlines(keepends=True)
            idx_for = next(
                (i for i, l in enumerate(lines) if "for batch in trainer.train_loader:" in l and not l.lstrip().startswith("#")),
                None,
            )
            if idx_for is None or idx_for + 1 >= len(lines):
                exit(1)

# Fix corrupted comment block if present (wrongly uncommented n_batches/train_loss in comment section).
for i in range(1, len(lines)):
    if "train_loss = total_loss.cpu().item() / n_batches" not in lines[i] or lines[i].lstrip().startswith("#"):
        continue
    if "n_batches = num_batches_done" not in lines[i - 1]:
        continue
    if i < 2 or not lines[i - 2].strip().startswith("#") or not lines[i - 2].startswith("    #"):
        continue
    lines[i - 1] = "    #             train_loss = total_loss.cpu().item() / len(trainer.train_loader)\n"
    del lines[i]
    s = "".join(lines)
    lines = s.splitlines(keepends=True)
    break

idx = idx_for

# Use exact leading whitespace from the file (preserves tabs vs spaces).
leading = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
pad = leading
inner = leading + ("\t" if "\t" in leading else "    ")
new_block = [
    pad + "max_batches = getattr(trainer.dataset_config, \"max_train_batches\", None)\n",
    pad + "num_batches_done = 0\n",
    lines[idx],
    inner + "if max_batches is not None and num_batches_done >= max_batches:\n",
    inner + "    break\n",
    inner + "self.optimizer.zero_grad()\n",
]
s = "".join(lines[:idx] + new_block + lines[idx + 2 :])

old = "                # if (global_step % ckpt_every_steps) == 0:\n                #     trainer.save_ckpt_last(epoch=epoch, extra={\"global_step\": global_step})\n\n\n            if self.scheduler"
new = "                # if (global_step % ckpt_every_steps) == 0:\n                #     trainer.save_ckpt_last(epoch=epoch, extra={\"global_step\": global_step})\n\n                num_batches_done += 1\n\n            if self.scheduler"
s = s.replace(old, new)

# Replace train_loss line only in active code (skip comment lines).
lines_out = s.splitlines(keepends=True)
for i, line in enumerate(lines_out):
    if "train_loss = total_loss.cpu().item() / len(trainer.train_loader)" not in line or line.lstrip().startswith("#"):
        continue
    indent = line[: len(line) - len(line.lstrip())]
    lines_out[i] = indent + "n_batches = num_batches_done if num_batches_done > 0 else 1\n" + indent + "train_loss = total_loss.cpu().item() / n_batches\n"
    break
s = "".join(lines_out)
path.write_text(s)
