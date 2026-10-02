#!/usr/bin/env python3
"""Idempotently register BACAM local inference tags in BFCL."""

from pathlib import Path
import sys

from bacam.paths import PATHS


CONFIG = Path(PATHS["BACAM_BFCL_ROOT"]) / "bfcl_eval/constants/model_config.py"
HANDLER_IMPORT = "from bfcl_eval.model_handler.local_inference.rlla import RLLAHandler\n"
BEGIN = "    # >>> BACAM BEGIN\n"
END = "    # <<< BACAM END\n"
STAGES = ("t2_tool", "t3_search", "t4_alfworld")
LABELS = ("r0", "r1", "r2")
TAGS = tuple(f"bacam-wtsa-{stage}-{label}" for stage in STAGES for label in LABELS) + tuple(
    f"bacam-wtsa-{model}-ours" for model in ("base", "t1", "t2", "t3", "t4",
                                                "search-expert", "tool-expert",
                                                "alfworld-expert", "webshop-expert"))
ENTRY = '''    "{name}": ModelConfig(
        model_name="{name}",
        display_name="BACAM: {name}",
        url="https://github.com/shuaitongli/BACAM",
        org="BACAM",
        license="apache-2.0",
        model_handler=RLLAHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    ),
'''


def map_close_line(lines: list[str]) -> int:
    start = next(index for index, line in enumerate(lines)
                 if line.startswith("local_inference_model_map = {"))
    return next(index for index in range(start + 1, len(lines)) if lines[index].startswith("}"))


def main() -> None:
    lines = CONFIG.read_text(encoding="utf-8").splitlines(keepends=True)
    if HANDLER_IMPORT not in lines:
        start = next(index for index, line in enumerate(lines)
                     if line.startswith("local_inference_model_map = {"))
        lines.insert(start, HANDLER_IMPORT + "\n")
    if BEGIN in lines:
        begin, end = lines.index(BEGIN), lines.index(END)
        del lines[begin:end + 1]
    block = [BEGIN] + [ENTRY.format(name=tag) for tag in TAGS] + [END]
    position = map_close_line(lines)
    lines[position:position] = block
    CONFIG.write_text("".join(lines), encoding="utf-8")
    print(f"registered {len(TAGS)} BACAM tags")


if __name__ == "__main__":
    main()
