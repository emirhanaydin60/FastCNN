import os
import json
import glob

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_val_loss(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "val_loss" in data and isinstance(data["val_loss"], list):
            return data["val_loss"]
        if "history" in data and isinstance(data["history"], dict) and "val_loss" in data["history"]:
            return data["history"]["val_loss"]
        for key in ("metrics",):
            if key in data and isinstance(data[key], dict) and "val_loss" in data[key]:
                return data[key]["val_loss"]

    raise ValueError(f"Could not find 'val_loss' list in {path}")


def model_name_from_filename(path):
    base = os.path.basename(path)
    name = base
    for suf in ("_history.json", ".json"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def load_models(hist_dir):
    pattern = os.path.join(hist_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSON history files found in {hist_dir}")

    models = {}
    for p in files:
        try:
            vl = load_val_loss(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        models[model_name_from_filename(p)] = vl

    if not models:
        raise RuntimeError("No valid histories loaded")

    return models


def analyze(hist_dir):
    models = load_models(hist_dir)
    max_epochs = max(len(v) for v in models.values())

    epoch_rankings = {}
    for epoch_idx in range(max_epochs):
        records = []
        for model, vals in models.items():
            val = vals[epoch_idx] if epoch_idx < len(vals) else None
            records.append({"model": model, "val_loss": val})

        def sort_key(r):
            return (r["val_loss"] is None, float("inf") if r["val_loss"] is None else r["val_loss"])

        records.sort(key=sort_key)
        epoch_rankings[f"epoch_{epoch_idx+1}"] = records

    return epoch_rankings


def compute_min_ranking(models):
    ranking = []
    for name, vals in models.items():
        if not vals:
            continue
        min_val = min(vals)
        epoch = vals.index(min_val) + 1
        ranking.append({"model": name, "min_val_loss": float(min_val), "epoch": int(epoch)})

    ranking.sort(key=lambda x: x["min_val_loss"])
    return ranking


def write_json(obj, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _pearson_corr(x, y):
    import math

    n = len(x)
    if n == 0:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return None
    return num / (sx * sy)


def _rankdata(a):
    # average ranks for ties, 1-based ranks
    n = len(a)
    if n == 0:
        return []
    # pair values with original indices
    enumerated = list(enumerate(a))
    # sort by value
    sorted_by_val = sorted(enumerated, key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # find group of equal values
        while j + 1 < n and sorted_by_val[j + 1][1] == sorted_by_val[i][1]:
            j += 1
        # average rank for positions i..j (1-based)
        rank_avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            orig_idx = sorted_by_val[k][0]
            ranks[orig_idx] = rank_avg
        i = j + 1
    return ranks


def spearman_corr(x, y):
    """Compute Spearman correlation (handles ties via average ranks)."""
    if not x or not y or len(x) != len(y):
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson_corr(rx, ry)


def compute_epoch_similarity(dataset, out_path=None, out_csv=None):
    hist_dir = os.path.join("results", dataset, "histories")
    models = load_models(hist_dir)

    ranked_path = os.path.join("results", dataset, "results_ranked.json")
    if not os.path.exists(ranked_path):
        raise FileNotFoundError(f"Final ranked results not found at {ranked_path}")
    with open(ranked_path, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    final_order = []
    for d in final_data:
        if "name" in d:
            final_order.append(d["name"])
        elif "model" in d:
            final_order.append(d["model"])

    final_order = [n for n in final_order if n in models]
    max_epochs = max(len(v) for v in models.values())

    summaries = []
    for epoch_idx in range(max_epochs):
        records = []
        for name in final_order:
            vals = models.get(name, [])
            val = vals[epoch_idx] if epoch_idx < len(vals) else None
            records.append({"model": name, "val_loss": val})

        def s_key(r):
            return (r["val_loss"] is None, float("inf") if r["val_loss"] is None else r["val_loss"])

        records_sorted = sorted(records, key=s_key)
        epoch_order = [r["model"] for r in records_sorted]

        epoch_ranks = {name: i + 1 for i, name in enumerate(epoch_order)}
        final_ranks = {name: i + 1 for i, name in enumerate(final_order)}

        x = [final_ranks[n] for n in final_order]
        y = [epoch_ranks[n] for n in final_order]
        corr_all = spearman_corr(x, y)

        top3 = final_order[:3]
        x3 = [final_ranks[n] for n in top3]
        y3 = [epoch_ranks.get(n, len(final_order) + 1) for n in top3]
        corr_top3 = spearman_corr(x3, y3)

        losses = [r["val_loss"] for r in records if r["val_loss"] is not None]
        epoch_min = min(losses) if losses else None

        summaries.append(
            {
                "epoch": epoch_idx + 1,
                "epoch_min_val_loss": epoch_min,
                "correlation_all": corr_all,
                "correlation_top3": corr_top3,
                "epoch_order": epoch_order,
                "epoch_vals": [{"model": r["model"], "val_loss": r["val_loss"]} for r in records_sorted],
            }
        )

    out_path = out_path or os.path.join("results", dataset, "epoch_similarity.json")
    write_json(summaries, out_path)

    if out_csv:
        import csv

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "epoch_min_val_loss", "correlation_all", "correlation_top3"])
            for s in summaries:
                w.writerow([s["epoch"], s["epoch_min_val_loss"], s["correlation_all"], s["correlation_top3"]])

    return summaries


def save_correlation_plots(dataset, summaries, out_all=None, out_top3=None):
    if plt is None:
        raise RuntimeError("matplotlib is required to produce correlation plots")

    epochs = [s["epoch"] for s in summaries]
    corr_all = [s.get("correlation_all") if s.get("correlation_all") is not None else float("nan") for s in summaries]
    corr_top3 = [s.get("correlation_top3") if s.get("correlation_top3") is not None else float("nan") for s in summaries]

    if out_all:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, corr_all, marker="o", linestyle="-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson on ranks (all models)")
        ax.set_title(f"{dataset} - Epoch similarity vs final ranking (all models)")
        ax.grid(True)
        os.makedirs(os.path.dirname(out_all), exist_ok=True)
        fig.savefig(out_all, dpi=200)
        plt.close(fig)

    if out_top3:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, corr_top3, marker="o", color="tab:orange", linestyle="-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pearson on ranks (top-3)")
        ax.set_title(f"{dataset} - Epoch similarity vs final ranking (top-3)")
        ax.grid(True)
        os.makedirs(os.path.dirname(out_top3), exist_ok=True)
        fig.savefig(out_top3, dpi=200)
        plt.close(fig)


def save_rank_names_table(dataset, summaries, final_ranking, out_csv=None, out_img=None, include_final_top=True):
    """Save a CSV and optional image table with model names per rank (no val_loss), plus min loss and correlations."""
    # final_ranking: list of model names in final order
    if not summaries:
        return

    num_models = len(summaries[0]["epoch_order"])
    # CSV
    if out_csv:
        import csv

        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["epoch"] + [f"rank_{i+1}" for i in range(num_models)] + ["epoch_min_val_loss", "correlation_all", "correlation_top3"]
            w.writerow(header)
            if include_final_top:
                row = ["final"] + final_ranking + ["", "", ""]
                w.writerow(row)
            for s in summaries:
                row = [f"epoch_{s['epoch']}"] + s["epoch_order"] + [s["epoch_min_val_loss"], s["correlation_all"], s["correlation_top3"]]
                w.writerow(row)

    # Image table
    if out_img:
        if plt is None:
            raise RuntimeError("matplotlib is required to produce image output")

        rows = []
        row_labels = []
        if include_final_top:
            rows.append(final_ranking)
            row_labels.append("final")
        for s in summaries:
            rows.append(s["epoch_order"])  # list of names
            row_labels.append(f"epoch {s['epoch']}")

        # add extra columns for min and correlations as separate columns at end
        # convert to strings
        extra_cols = [["" for _ in rows] for _ in range(3)]
        for i, s in enumerate(summaries):
            idx = i + (1 if include_final_top else 0)
            extra_cols[0][idx] = f"{s['epoch_min_val_loss']:.4f}" if s["epoch_min_val_loss"] is not None else ""
            extra_cols[1][idx] = f"{s['correlation_all']:.4f}" if s["correlation_all"] is not None else ""
            extra_cols[2][idx] = f"{s['correlation_top3']:.4f}" if s["correlation_top3"] is not None else ""

        # build table cell text
        cell_text = []
        for r_idx, row in enumerate(rows):
            cells = []
            for name in row:
                cells.append(name)
            # append extras
            cells.append(extra_cols[0][r_idx])
            cells.append(extra_cols[1][r_idx])
            cells.append(extra_cols[2][r_idx])
            cell_text.append(cells)

        col_labels = [f"Rank {i+1}" for i in range(num_models)] + ["min_val", "corr_all", "corr_top3"]

        # draw
        fig_h = max(4, 0.35 * len(cell_text) + 1.5)
        fig_w = max(8, 1.2 * num_models)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        table = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        fig.savefig(out_img, dpi=200)
        plt.close(fig)


def save_table_image(rankings, out_path, top_k=None):
    if plt is None:
        raise RuntimeError("matplotlib is required to produce image output")

    epochs = sorted(rankings.keys(), key=lambda x: int(x.split("_")[1]))
    num_epochs = len(epochs)
    first = rankings[epochs[0]]
    num_models = len(first)
    cols = num_models if top_k is None else min(top_k, num_models)

    cell_text = []
    for e in epochs:
        row = []
        for r in rankings[e][:cols]:
            if r["val_loss"] is None:
                row.append(f"{r['model']}\n-")
            else:
                row.append(f"{r['model']}\n{r['val_loss']:.4f}")
        cell_text.append(row)

    col_labels = [f"Rank {i+1}" for i in range(cols)]
    row_labels = [f"Epoch {i+1}" for i in range(num_epochs)]

    col_w = 2.5
    row_h = 0.4
    fig_w = max(6, col_w * cols)
    fig_h = max(4, row_h * num_epochs + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_csv(rankings, out_path, top_k=None):
    import csv

    epochs = sorted(rankings.keys(), key=lambda x: int(x.split("_")[1]))
    first = rankings[epochs[0]]
    num_models = len(first)
    cols = num_models if top_k is None else min(top_k, num_models)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["epoch"] + [f"rank_{i+1}" for i in range(cols)]
        w.writerow(header)
        for e in epochs:
            row = [e]
            for r in rankings[e][:cols]:
                if r["val_loss"] is None:
                    row.append(f"{r['model']}|-")
                else:
                    row.append(f"{r['model']}|{r['val_loss']:.4f}")
            w.writerow(row)


def main(dataset="BT", input_dir=None, output_json=None, topk=None, save_json_flag=True, save_img_flag=True, save_csv_flag=True):
    base = input_dir if input_dir is not None else os.path.join("results", dataset, "histories")
    out_json = output_json if output_json is not None else os.path.join("results", dataset, "epoch_rankings.json")
    out_img = os.path.join("results", dataset, "epoch_rankings_table.png")
    out_csv = os.path.join("results", dataset, "epoch_rankings_table.csv")

    rankings = analyze(base)

    models = load_models(base)
    min_ranking = compute_min_ranking(models)
    out_ranked = os.path.join("results", dataset, "results_ranked.json")
    write_json(min_ranking, out_ranked)
    print(f"Wrote final ranked results to {out_ranked}")

    if save_json_flag:
        write_json(rankings, out_json)
        print(f"Wrote rankings JSON to {out_json}")

    if save_csv_flag:
        save_csv(rankings, out_csv, top_k=topk)
        print(f"Wrote CSV summary to {out_csv}")

    if save_img_flag:
        try:
            save_table_image(rankings, out_img, top_k=topk)
            print(f"Wrote table image to {out_img}")
        except Exception as e:
            print(f"Could not create image: {e}")

    sim_csv = os.path.join("results", dataset, "epoch_similarity.csv")
    try:
        summaries = compute_epoch_similarity(dataset, out_csv=sim_csv)
        print(f"Wrote epoch similarity JSON to results/{dataset}/epoch_similarity.json and CSV to {sim_csv}")
        out_corr_all = os.path.join("results", dataset, "correlation_all.png")
        out_corr_top3 = os.path.join("results", dataset, "correlation_top3.png")
        try:
            save_correlation_plots(dataset, summaries, out_all=out_corr_all, out_top3=out_corr_top3)
            print(f"Wrote correlation plots to {out_corr_all} and {out_corr_top3}")
        except Exception as e:
            print(f"Could not create correlation plots: {e}")

        corr_list = [s for s in summaries if s.get("correlation_all") is not None]
        if corr_list:
            corr_list.sort(key=lambda s: s["correlation_all"], reverse=True)
            print("Top epochs by correlation_all:")
            for s in corr_list[:5]:
                print(f" Epoch {s['epoch']}: corr_all={s['correlation_all']:.4f}, corr_top3={s['correlation_top3']}, min_val={s['epoch_min_val_loss']}")
    except Exception as e:
        print(f"Could not compute epoch similarity: {e}")
    # create rank-names-only table
    try:
        final_ranking = [d["model"] for d in min_ranking]
        out_rank_names_csv = os.path.join("results", dataset, "epoch_rank_names_table.csv")
        out_rank_names_img = os.path.join("results", dataset, "epoch_rank_names_table.png")
        save_rank_names_table(dataset, summaries, final_ranking, out_csv=out_rank_names_csv, out_img=out_rank_names_img, include_final_top=True)
        print(f"Wrote epoch rank-names table to {out_rank_names_csv} and {out_rank_names_img}")
    except Exception as e:
        print(f"Could not write rank-names table: {e}")


if __name__ == "__main__":
    main("MNIST")
