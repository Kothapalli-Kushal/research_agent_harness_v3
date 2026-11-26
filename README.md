# ML Research Harness v3

Run bounded, agent-led ML studies and export a fitted model. This repo ports the harness from UW Math AI's repository at commit `6b797a4`. V3 uses registered comparisons, development evidence, and an independent critic. It selects a model on validation data, refits it, and certifies the serialized model before scoring it on held-out data. [Upstream harness](https://github.com/uw-math-ai/ai-inverse-lab/tree/6b797a464ed625dd072011f342e2865cd2a5654d/ML%20Agent%20Building/Automated%20ML%20Research%20Harness)

## Run with Astra

From the repository root, in a Python 3.11+ environment, install the CPU dependencies and sign in to a current Codex CLI with your ChatGPT account.

```bash
python -m pip install "numpy>=1.24" "scipy>=1.10" "scikit-learn>=1.3" "pyyaml>=6.0"
codex login
codex login status
python -m mlharness research --pipeline v3 --spec specs/v3_example.yaml \
  --backend codex --model gpt-6.2-astra --run-dir runs/heat-astra
```

Codex uses your ChatGPT plan when signed in with your ChatGPT account. Astra guides the research decisions, and the harness trains candidate estimators on your machine. The example's roles inherit the selected backend and model. For a separate research problem, copy the spec and set its simulator or data source, metrics, study arms, and compute and LLM call limits. [Codex plan usage](https://help.openai.com/en/articles/20001275-chatgpt-work-and-codex)

For an offline run, use `--backend none --run-dir runs/heat-offline` in place of the Astra options. The Codex backend logs service errors and falls back to the harness's offline policy within the run. Check `artifacts.jsonl` for the model and outcome of each role call. The example permits one final evaluation per held-out split in the local exposure registry. Resume an interrupted run with the same spec, options, and run directory plus `--resume`.

## Load the model

Each successful V3 command prints the path to `final/model.pkl`. Keep the full `final/` directory when moving an export so its manifest and any custom model or simulator code travel with it. Load only exports you trust because pickle can execute code.

```bash
python -m mlharness predict runs/heat-astra \
  --input runs/heat-astra/data/dataset.npz --key X_test --output predictions.npy
```

The included [example model](examples/heat_v3_model/model.pkl) is a fitted ridge regressor trained offline on simulated heat traces. It maps 40 time-major temperature readings to two conductivities. The [manifest](examples/heat_v3_model/manifest.json) records the dependency versions, certification checks, and held-out relative L2 error of 0.00995.

```bash
python - <<'PY'
from examples.heat_diffusion_simulator import simulate
from mlharness.export import load_model

X, _ = simulate(3, seed=10, n_segments=2, n_grid=20, n_steps=60,
                sensors=(3, 8, 12, 17), every=6)
print(load_model("examples/heat_v3_model").predict(X))
PY
```

See the [V3 usage guide](Version%20Builds/v3/V3_USAGE.md) for role configuration, budgets, and evidence. The [architecture guide](Version%20Builds/v3/V3_ARCHITECTURE.md) documents the execution and export contracts. `requirements.txt` also lists the optional Torch, Anthropic, SSH, and HTTP packages.
