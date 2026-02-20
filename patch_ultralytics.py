# One-time patcher for Ultralytics to reproduce PCG-YOLO training.
# (1) copy custom modules into Ultralytics
# (2) patch parse_model sets to register C3CACG
# (3) patch bbox IoU term to PIoUv2 (by replacing bbox_iou(...CIoU=True) call)

from __future__ import annotations

import re
import shutil
from pathlib import Path

import ultralytics


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")


def _backup_once(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists() and p.exists():
        shutil.copy2(p, bak)
        print(f"[OK] Backup created: {bak}")


def _ensure_line(text: str, line: str) -> str:
    if line in text:
        return text
    if not text.endswith("\n"):
        text += "\n"
    return text + line + "\n"


def _insert_import_after_import_block(src: str, import_line: str) -> str:
    """Insert an import line after the initial import block (safe for runtime)."""
    if import_line in src:
        return src

    lines = src.splitlines(True)

    # Skip shebang / encoding comments
    i = 0
    while i < len(lines) and (lines[i].startswith("#!") or "coding:" in lines[i]):
        i += 1

    # Skip module docstring if present
    if i < len(lines) and lines[i].lstrip().startswith(('"""', "'''")):
        q = lines[i].lstrip()[:3]
        i += 1
        while i < len(lines) and q not in lines[i]:
            i += 1
        if i < len(lines):
            i += 1

    # Now i is after docstring; advance through consecutive import lines (and blank/comment lines inside import block)
    j = i
    while j < len(lines):
        s = lines[j].strip()
        if s == "" or s.startswith("#"):
            j += 1
            continue
        if s.startswith("import ") or s.startswith("from "):
            j += 1
            continue
        break

    # Insert import_line + newline at position j
    lines.insert(j, import_line + "\n")
    return "".join(lines)


def patch_ultralytics():
    repo_root = Path(__file__).resolve().parent
    ulta_root = Path(ultralytics.__file__).resolve().parent
    print(f"[INFO] Ultralytics path: {ulta_root}")

    # --- 1) copy custom nn modules ---
    src_mod = repo_root / "trackyolo" / "ultra_modules.py"
    dst_mod_dir = ulta_root / "nn" / "modules"
    dst_mod_dir.mkdir(parents=True, exist_ok=True)
    dst_mod = dst_mod_dir / "trackyolo.py"
    shutil.copy2(src_mod, dst_mod)
    print(f"[OK] Copied custom modules -> {dst_mod}")

    # Ensure ultralytics.nn.modules.__init__ imports C3CACG
    init_py = dst_mod_dir / "__init__.py"
    init_py.touch(exist_ok=True)
    init_txt = _read(init_py)
    init_txt = _ensure_line(init_txt, "from .trackyolo import C3CACG  # PCG-YOLO custom")
    _write(init_py, init_txt)
    print("[OK] Patched nn/modules/__init__.py import")

    # --- 1b) copy PIoU helper into Ultralytics (self-contained) ---
    src_piou = repo_root / "trackyolo" / "piou.py"
    dst_piou = ulta_root / "utils" / "pcg_piou.py"
    shutil.copy2(src_piou, dst_piou)
    print(f"[OK] Copied PIoU helper -> {dst_piou}")

    # --- 2) patch nn/tasks.py sets (base_modules / repeat_modules) ---
    tasks_py = ulta_root / "nn" / "tasks.py"
    _backup_once(tasks_py)
    t = _read(tasks_py)

    if "C3CACG" not in t:
        # Try to inject C3CACG into base_modules and repeat_modules frozensets
        t_new = t
        t_new, n1 = re.subn(
            r"(base_modules\s*=\s*frozenset\(\s*\{\s*)",
            r"\1C3CACG,  # PCG-YOLO custom\n        ",
            t_new,
            count=1,
        )
        t_new, n2 = re.subn(
            r"(repeat_modules\s*=\s*frozenset\(\s*# modules with 'repeat' arguments\s*\{\s*)",
            r"\1C3CACG,  # PCG-YOLO custom\n        ",
            t_new,
            count=1,
        )

        if n1 == 1 and n2 == 1:
            _write(tasks_py, t_new)
            print("[OK] Patched nn/tasks.py (base_modules/repeat_modules)")
        else:
            print("[WARN] Could not patch nn/tasks.py automatically (Ultralytics version/layout mismatch).")
            print("       Please add C3CACG manually into base_modules and repeat_modules in ultralytics/nn/tasks.py.")
    else:
        print("[SKIP] nn/tasks.py already contains C3CACG")

    # --- 3) patch utils/loss.py: replace bbox_iou(...CIoU=True) call with PIoUv2 loss proxy ---
    loss_py = ulta_root / "utils" / "loss.py"
    _backup_once(loss_py)
    loss_txt = _read(loss_py)

    # Insert import near the top (before use)
    loss_txt = _insert_import_after_import_block(
        loss_txt,
        "from .pcg_piou import piou  # PIoUv2 (PCG-YOLO)",
    )

    # Replace the bbox_iou call (keep the variable name `iou` unchanged)
    patt = (
        r"bbox_iou\(\s*pred_bboxes\[fg_mask\]\s*,\s*target_bboxes\[fg_mask\]\s*,\s*xywh=False\s*,\s*CIoU=True\s*\)"
    )
    repl = "1 - piou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIoU2=True)"

    if re.search(patt, loss_txt) and "PIoU2=True" not in loss_txt:
        loss_txt = re.sub(patt, repl, loss_txt, count=1)
        _write(loss_py, loss_txt)
        print("[OK] Switched bbox IoU term -> PIoUv2 proxy in utils/loss.py")
    else:
        _write(loss_py, loss_txt)
        print("[WARN] Could not find target bbox_iou(...CIoU=True) pattern OR already patched.")
        print("       Please check ultralytics/utils/loss.py manually to ensure PIoUv2 is actually used.")

    print("[DONE] Patch finished.")


if __name__ == "__main__":
    patch_ultralytics()