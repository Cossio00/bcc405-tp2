import subprocess

models = ["mlp", "cnn"]
attacks = ["dlg", "idlg"]
defenses = ["none", "clipping", "noise"]
opts = ["L-BFGS", "Adam"]
scenarios = ["single", "batch"]

for model in models:
    for attack in attacks:
        for defense in defenses:
            for opt in opts:
                for scenario in scenarios:
                    cmd = [
                        "python", "-m", "runner.invert",
                        "--model", model,
                        "--attack", attack,
                        "--defense", defense,
                        "--opt", opt,
                        "--scenario", scenario,
                        "--iters", "500",       # pode ajustar
                        "--restarts", "2",      # pode ajustar
                        "--lr", "0.1"
                    ]
                    if defense == "clipping":
                        cmd += ["--clip", "0.5"]
                    if defense == "noise":
                        cmd += ["--sigma", "1e-3"]

                    print("\nRodando:", " ".join(cmd))
                    subprocess.run(cmd)
