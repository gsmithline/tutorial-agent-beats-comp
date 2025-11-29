import os
import sys
import argparse
import json
import time
import glob
import shutil
import concurrent.futures
from typing import Dict, List, Tuple, Optional

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.negotiation_game import run_game


def _default_model_registry() -> Dict[str, List[int]]:
    """
    Registry of available agents (model name -> list of circles).
    Adjust here or pass a custom registry into run_matrix_pipeline.
    """
    return {
        "tough_agent": [0],
        "conceder_agent": [0],
    }


def _default_shortnames(model_circles: Dict[str, List[int]]) -> Dict[str, str]:
    """Create shortnames for folder naming; override if needed."""
    return {model: model for model in model_circles.keys()}


def _build_experiments(
    model_circles: Dict[str, List[int]],
    full_matrix: bool,
    model: Optional[str],
    circle: Optional[int],
) -> List[Dict]:
    """Create the list of experiment specs (p1_model/p2_model/circles)."""
    all_combinations = []
    for model_name, circles in model_circles.items():
        for c in circles:
            all_combinations.append((model_name, c))

    experiments: List[Dict] = []

    if full_matrix:
        for p1_model, p1_circle in all_combinations:
            for p2_model, p2_circle in all_combinations:
                experiments.append(
                    {
                        "p1_model": p1_model,
                        "p1_circle": p1_circle,
                        "p2_model": p2_model,
                        "p2_circle": p2_circle,
                    }
                )
    else:
        if model is None or circle is None:
            raise ValueError(
                "Either --full_matrix must be specified or both --model and --circle must be provided"
            )

        if model not in model_circles:
            raise ValueError(f"Invalid model: {model}. Valid models are: {list(model_circles.keys())}")

        if circle not in model_circles[model]:
            raise ValueError(
                f"Circle {circle} is not valid for model {model}. Valid circles are: {model_circles[model]}"
            )

        for opponent_model, opponent_circle in all_combinations:
            experiments.append(
                {
                    "p1_model": model,
                    "p1_circle": circle,
                    "p2_model": opponent_model,
                    "p2_circle": opponent_circle,
                }
            )

            if not (opponent_model == model and opponent_circle == circle):
                experiments.append(
                    {
                        "p1_model": opponent_model,
                        "p1_circle": opponent_circle,
                        "p2_model": model,
                        "p2_circle": circle,
                    }
                )

    return experiments


def _attach_output_dirs(
    experiments: List[Dict],
    base_dir: str,
    model_shortnames: Dict[str, str],
    date: str,
    skip_existing: bool,
    force_new_dirs: bool,
    dry_run: bool,
    debug: bool,
) -> List[Dict]:
    """Populate each experiment with output_dir/prompt_style/skip flags."""
    shortname_to_model = {v: k for k, v in model_shortnames.items()}
    existing_dirs = {}

    if os.path.exists(base_dir) and not force_new_dirs:
        all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        for dir_name in all_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            parts = dir_name.split("_")

            found_models = []
            found_circles = []

            for i, part in enumerate(parts):
                if part in shortname_to_model:
                    found_models.append(part)
                elif "circle" in part and i + 1 < len(parts) and parts[i + 1].isdigit():
                    found_circles.append(int(parts[i + 1]))
                elif part.isdigit() and i > 0 and "circle" in parts[i - 1]:
                    continue

            if len(found_models) == 2 and len(found_circles) == 2:
                key = (
                    (shortname_to_model[found_models[0]], found_circles[0]),
                    (shortname_to_model[found_models[1]], found_circles[1]),
                )

                existing_dirs[key] = dir_path

                if debug:
                    print(f"[DEBUG] Mapped directory {dir_name} to:")
                    print(f"        - {found_models[0]}_{found_circles[0]} vs {found_models[1]}_{found_circles[1]}")

    for exp in experiments:
        key = ((exp["p1_model"], exp["p1_circle"]), (exp["p2_model"], exp["p2_circle"]))

        if key in existing_dirs and not force_new_dirs:
            exp["output_dir"] = existing_dirs[key]
            exp["found_existing"] = True
            if debug:
                print(
                    f"[DEBUG] Found existing directory for {model_shortnames[exp['p1_model']]}_{exp['p1_circle']} "
                    f"vs {model_shortnames[exp['p2_model']]}_{exp['p2_circle']}: {exp['output_dir']}"
                )
        else:
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            p1_short = model_shortnames[exp["p1_model"]]
            p2_short = model_shortnames[exp["p2_model"]]
            dir_name = f"{p1_short}_{p2_short}_circle_{exp['p1_circle']}_vs_circle_{exp['p2_circle']}"

            exp["output_dir"] = os.path.join(base_dir, dir_name)
            exp["found_existing"] = False

            if not dry_run:
                os.makedirs(exp["output_dir"], exist_ok=True)
                print(f"[INFO] Created new directory: {exp['output_dir']}")
            else:
                print(f"[INFO] Would create new directory: {exp['output_dir']}")

        folder_name = os.path.basename(exp["output_dir"])
        exp["prompt_style"] = folder_name

        if skip_existing:
            existing_files = glob.glob(os.path.join(exp["output_dir"], f"*{date}*.json")) + glob.glob(
                os.path.join(exp["output_dir"], f"*{date}*.pkl")
            )

            exp["skip"] = bool(existing_files)
            if exp["skip"]:
                print(f"[INFO] Skipping {exp['prompt_style']} - data already exists for date {date}")
        else:
            exp["skip"] = False

    return experiments


def _find_and_move_files(exp: Dict, date: str, model_shortnames: Dict[str, str]) -> int:
    """Locate generated JSON/PKL files and move them into the experiment output_dir."""
    p1_model_short = model_shortnames[exp["p1_model"]]
    p2_model_short = model_shortnames[exp["p2_model"]]
    p1_circle = exp["p1_circle"]
    p2_circle = exp["p2_circle"]

    pattern_bases = [
        f"*{p1_model_short}*{p2_model_short}*circle_p1_{p1_circle}*circle_p2_{p2_circle}*{date}*",
        f"*{p1_model_short}*{p2_model_short}*{p1_circle}*{p2_circle}*{date}*",
        f"*{exp['prompt_style']}*{date}*",
        f"*{date}*{p1_model_short}*{p2_model_short}*circle*{p1_circle}*{p2_circle}*",
    ]

    search_locations = [".", "output", "results", "data"]
    found_files = []

    for location in search_locations:
        if os.path.exists(location):
            for pattern_base in pattern_bases:
                json_pattern = os.path.join(location, pattern_base + ".json")
                pkl_pattern = os.path.join(location, pattern_base + ".pkl")

                found_files.extend(glob.glob(json_pattern))
                found_files.extend(glob.glob(pkl_pattern))

    moved_count = 0
    for file_path in found_files:
        dest_path = os.path.join(exp["output_dir"], os.path.basename(file_path))
        if os.path.exists(file_path) and file_path != dest_path:
            shutil.move(file_path, dest_path)
            print(f"[INFO] Moved {os.path.basename(file_path)} to {exp['output_dir']}")
            moved_count += 1

    return moved_count


def run_matrix_pipeline(
    model_circles: Optional[Dict[str, List[int]]] = None,
    model_shortnames: Optional[Dict[str, str]] = None,
    *,
    full_matrix: bool = True,
    matrix_id: str | int = 1,
    model: Optional[str] = None,
    circle: Optional[int] = None,
    date: Optional[str] = None,
    max_rounds: int = 5,
    games: int = 20,
    parallel: bool = True,
    discount: float = 0.98,
    skip_existing: bool = False,
    force_new_dirs: bool = False,
    dry_run: bool = False,
    use_openspiel: bool = True,
    num_items: int = 3,
    debug: bool = False,
) -> Dict:
    """
    Programmatic entry point: run the full cross-play matrix of OpenSpiel games.

    Returns a dict with experiments and the base output directory for downstream meta-game analysis.
    """
    model_circles = model_circles or _default_model_registry()
    model_shortnames = model_shortnames or _default_shortnames(model_circles)
    date = date or time.strftime("%m_%d_%Y")

    experiments = _build_experiments(model_circles, full_matrix, model, circle)

    base_dir = f"crossplay/game_matrix_{matrix_id}"
    experiments = _attach_output_dirs(
        experiments,
        base_dir=base_dir,
        model_shortnames=model_shortnames,
        date=date,
        skip_existing=skip_existing,
        force_new_dirs=force_new_dirs,
        dry_run=dry_run,
        debug=debug,
    )

    orig_count = len(experiments)
    experiments = [exp for exp in experiments if not exp.get("skip", False)]
    skipped_count = orig_count - len(experiments)

    if skipped_count > 0:
        print(f"[INFO] Skipped {skipped_count} experiments that already have data")

    model_pairs = set()
    for exp in experiments:
        pair = f"{model_shortnames[exp['p1_model']]}_{exp['p1_circle']} vs {model_shortnames[exp['p2_model']]}_{exp['p2_circle']}"
        model_pairs.add(pair)

    print(f"[INFO] Starting {'FULL MATRIX' if full_matrix else 'TARGETED'} experiments")
    print(f"  Total experiments to run: {len(experiments)}")
    print(f"  Unique model pairings: {len(model_pairs)}")
    print(f"  Date: {date}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Games per pairing: {games}")
    print(f"  Discount: {discount}")
    print(f"  Matrix ID: {matrix_id}")
    print(f"  Parallel: {parallel}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Dry run: {dry_run}")
    print(f"  Force new dirs: {force_new_dirs}")
    print(f"  Debug: {debug}")
    print(f"  Model: {model}")
    print(f"  Circle: {circle}")
    print(f"  Full matrix: {full_matrix}")
    print(f"  Openspiel: {use_openspiel}")
    print(f"  Number of items: {num_items}")
    print("--------------------------------------------------")
    for pair in sorted(model_pairs):
        print(f"  - {pair}")
    print("--------------------------------------------------")

    if dry_run:
        print("[INFO] Dry run complete. No experiments were actually run.")
        return {"experiments": experiments, "base_dir": base_dir}

    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {}
            completed_count = 0
            total_count = len(experiments)

            for exp in experiments:
                future = executor.submit(
                    run_game,
                    exp["p1_circle"],
                    exp["p2_circle"],
                    games,
                    max_rounds,
                    date,
                    exp["prompt_style"],
                    exp["p1_model"],
                    exp["p2_model"],
                    discount,
                    use_openspiel,
                    num_items,
                )
                futures[future] = exp
                print(f"[INFO] Submitted: {exp['prompt_style']} ({len(futures)}/{total_count})")

            for future in concurrent.futures.as_completed(futures):
                exp = futures[future]
                try:
                    future.result()
                    completed_count += 1
                    print(f"[INFO] Completed: {exp['prompt_style']} ({completed_count}/{total_count})")
                    moved = _find_and_move_files(exp, date, model_shortnames)
                    print(f"[INFO] Moved {moved} files to {exp['output_dir']}")
                except Exception as exc:
                    print(f"[ERROR] {exp['prompt_style']} generated exception: {exc}")

    else:
        completed_count = 0
        total_count = len(experiments)
        for exp in experiments:
            try:
                print(f"[INFO] Starting: {exp['prompt_style']} ({completed_count+1}/{total_count})")
                run_game(
                    exp["p1_circle"],
                    exp["p2_circle"],
                    games,
                    max_rounds,
                    date,
                    exp["prompt_style"],
                    exp["p1_model"],
                    exp["p2_model"],
                    discount,
                    use_openspiel,
                    num_items,
                )
                completed_count += 1
                print(f"[INFO] Completed: {exp['prompt_style']} ({completed_count}/{total_count})")
                moved = _find_and_move_files(exp, date, model_shortnames)
                print(f"[INFO] Moved {moved} files to {exp['output_dir']}")
            except Exception as exc:
                print(f"[ERROR] {exp['prompt_style']} generated exception: {exc}")

    print("[INFO] All experiment runs completed.")
    return {"experiments": experiments, "base_dir": base_dir}

def main():
    parser = argparse.ArgumentParser(description="Run negotiation experiments for models in the game matrix.")
    parser.add_argument("--model", type=str, required=False, help="Single model to run (use with --circle)")
    parser.add_argument("--circle", type=int, required=False, help="Circle value for the model")
    parser.add_argument("--full_matrix", action="store_true", help="Run the entire game matrix (all combinations)")
    parser.add_argument("--date", type=str, required=False, default=time.strftime('%m_%d_%Y'), help="Date string for output naming")
    parser.add_argument("--max_rounds", type=int, required=False, default=5, help="Maximum number of negotiation rounds")
    parser.add_argument("--games", type=int, required=False, default=20, help="Number of games to run per pairing")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run experiments in parallel")
    parser.add_argument("--discount", type=float, required=False, default=.98, help="Discount rate for game (between 0 and 1)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip experiments that already have data in their directories")
    parser.add_argument("--matrix_id", required=False, default=1, help="ID of the game matrix to use")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be run without actually running experiments")
    parser.add_argument("--force_new_dirs", action="store_true", help="Force creation of new directories for each experiment even if similar ones exist")
    parser.add_argument("--use_openspiel", action="store_true", default=True, help="Use OpenSpiel for running the experiments instead of custom implementation.")
    parser.add_argument("--num_items", type=int, default=3, help="Number of items in the negotiation game.")
    args = parser.parse_args()

    run_matrix_pipeline(
        full_matrix=args.full_matrix,
        matrix_id=args.matrix_id,
        model=args.model,
        circle=args.circle,
        date=args.date,
        max_rounds=args.max_rounds,
        games=args.games,
        parallel=args.parallel,
        discount=args.discount,
        skip_existing=args.skip_existing,
        force_new_dirs=args.force_new_dirs,
        dry_run=args.dry_run,
        use_openspiel=args.use_openspiel,
        num_items=args.num_items,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
