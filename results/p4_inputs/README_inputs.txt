p4_inputs: what P4 needs from the finished P1/P2/P3 runs
manifest.json         paths + hashes of the frozen qa_dev/qa_test (P4 must use these exact files), run folders, prefix.pt path
training_summary.json steps, sec/step (also stage C only), stage start steps, best/last val, loss components, IoU trajectory per run
per_template.csv      accuracy per model/split/template
weak_types.json       stage-C sampling weights (dev-only) + weak-template diagnostics (dev+test pooled)
zoom_stats.json       hints, IoU, center-hit per kind
reference_table.md    P0-P3 totals to paste next to P4
p1_vs_p2_paired.json  discordant pairs for the McNemar baseline
p4_cli_flags.txt      VLM flags of the P2 prefix; P4 reuses them unchanged
p4_budget.json        step counts and GPU-time projection for P4 and the compute-matched P2 control
