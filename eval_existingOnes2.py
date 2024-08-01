import os
import argparse
from glob import glob
import prettytable as pt

from evaluation.evaluate import evaluator
from config import Config


config = Config()


def do_eval(args):

    # Assuming args.pred_root and args.gt_root are defined and contain the paths to the directories
    def filter_image_files(file_list):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return [
            file
            for file in file_list
            if os.path.splitext(file)[1].lower() in image_extensions
        ]

    # Assuming args.pred_root and args.gt_root are defined and contain the paths to the directories
    pred_data_dir = sorted(glob(os.path.join(args.pred_root, "*")))
    gt_paths = sorted(glob(os.path.join(args.gt_root, "*")))

    # Filter to keep only image files
    pred_data_dir = filter_image_files(pred_data_dir)
    gt_paths = filter_image_files(gt_paths)

    # Extract base names without extensions
    pred_basenames = {
        os.path.splitext(os.path.basename(path))[0] for path in pred_data_dir
    }
    gt_basenames = {os.path.splitext(os.path.basename(path))[0] for path in gt_paths}

    # Filter lists to keep only items with matching base names
    filtered_pred_data_dir = [
        path
        for path in pred_data_dir
        if os.path.splitext(os.path.basename(path))[0] in gt_basenames
    ]
    filtered_gt_paths = [
        path
        for path in gt_paths
        if os.path.splitext(os.path.basename(path))[0] in pred_basenames
    ]

    filename = f"{args.model}.txt"
    tb = pt.PrettyTable()
    tb.vertical_char = "&"

    tb.field_names = [
        "Dataset",
        "Method",
        "Smeasure",
        "MAE",
        "maxEm",
        "meanEm",
        "maxFm",
        "meanFm",
        "wFmeasure",
        "adpEm",
        "adpFm",
        "HCE",
    ]

    em, sm, fm, mae, wfm, hce = evaluator(
        gt_paths=filtered_gt_paths,
        pred_paths=filtered_pred_data_dir,
        metrics=args.metrics.split("+"),
        verbose=config.verbose_eval,
    )

    scores = [
        sm.round(3),
        mae.round(3),
        em["curve"].max().round(3),
        em["curve"].mean().round(3),
        fm["curve"].max().round(3),
        fm["curve"].mean().round(3),
        wfm.round(3),
        em["adp"].round(3),
        fm["adp"].round(3),
        int(hce.round()),
    ]

    for idx_score, score in enumerate(scores):
        scores[idx_score] = (
            "." + format(score, ".3f").split(".")[-1]
            if score <= 1
            else format(score, "<4")
        )
    records = ["Humans", args.model] + scores
    tb.add_row(records)
    # Write results after every check.
    with open(filename, "w+") as file_to_write:
        file_to_write.write(str(tb) + "\n")
    print(tb)


if __name__ == "__main__":
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_root",
        type=str,
        help="ground-truth root",
        default=os.path.join(config.data_root_dir, config.task),
    )
    parser.add_argument(
        "--pred_root", type=str, help="prediction root", default="./e_preds"
    )
    parser.add_argument("--model", type=str, help="model", default="chkpt/epoch_56.pth")
    parser.add_argument(
        "--data_lst",
        type=str,
        help="test dataset",
        default={
            "humans": "+".join(["validation"][:]),
            "DIS5K": "+".join(
                ["DIS-VD", "DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"][:]
            ),
            "COD": "+".join(["TE-COD10K", "NC4K", "TE-CAMO", "CHAMELEON"][:]),
            "HRSOD": "+".join(
                ["DAVIS-S", "TE-HRSOD", "TE-UHRSD", "TE-DUTS", "DUT-OMRON"][:]
            ),
            "General": "+".join(["DIS-VD"][:]),
            "Portrait": "+".join(["TE-P3M-500-P"][:]),
        }[config.task],
    )
    parser.add_argument(
        "--check_integrity",
        type=bool,
        help="whether to check the file integrity",
        default=False,
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="candidate competitors",
        default="+".join(
            ["S", "MAE", "E", "F", "WF", "HCE"][: 100 if "DIS5K" in config.task else -1]
        ),
    )
    args = parser.parse_args()

    # start engine
    do_eval(args)
