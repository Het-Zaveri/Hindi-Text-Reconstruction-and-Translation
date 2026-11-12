# Windows installation and reproducible commands

Run PowerShell commands from the repository root. The commands shown use Windows paths; WSL, Docker, and Colab are not required.

## Python environments in the working project

The workstation has Python 3.12.10 and 3.14.7. The existing project `venv` (Python 3.14.7) was used for the most recent 65-test regression run. The dedicated `.venv_model3_indictrans2` environment (Python 3.12.10) holds the isolated CPU translation dependencies. `.venv_model3` belongs to the earlier rejected M2M100 experiment; it is not needed for Model 1, Model 2, or the final IndicTrans2 documentation.

The commands below create a separate environment for a new checkout. They do not recreate any existing local environment.

## Core environment (Models 1 and 2)

```powershell
git clone https://github.com/Het-Zaveri/Hindi-Text-Reconstruction-and-Translation.git
Set-Location Hindi-Text-Reconstruction-and-Translation

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The root `requirements.txt` is shared by the core code. It is separate from translation experiment dependency snapshots. Model weights and the Model 1 helper are fetched from Hugging Face when their scripts run. The installation sequence has not been validated as a clean-clone install on every Windows/Python combination; the documented results used the environments above.

## Run Model 1

`src/run_on_dataset.py` has no CLI flags. It reads `data/*.txt` and writes `outputs/<sample>_punctuated.txt`.

```powershell
.\.venv\Scripts\python.exe src\run_on_dataset.py
```

To reuse cached punctuation and evaluate the existing cached Model 1 predictions without forcing fresh model inference:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate.py --skip-inference
```

This evaluation command recomputes metrics and writes evaluation report files. `--force-inference` downloads/loads the model and refreshes the prediction cache; use it only when intentionally rerunning that evaluation.

## Run Model 2 or the complete reconstruction pipeline

After Model 1 has written punctuated files:

```powershell
# Selected practical MiniLM method on all seven punctuated inputs
.\.venv\Scripts\python.exe src\model2_ordering.py --method minilm

# Run Model 1 followed by MiniLM ordering from the raw inputs
.\.venv\Scripts\python.exe src\pipeline.py --method minilm
```

Both command forms were checked against the scripts' `--help` output. Model 2's default is `compare`, which also runs the optional mBERT NSP experiment. `--method minilm` avoids that comparison. `src/pipeline.py --skip-punctuation` reuses existing `outputs/<sample>_punctuated.txt` files.

Useful help commands:

```powershell
.\.venv\Scripts\python.exe src\model2_ordering.py --help
.\.venv\Scripts\python.exe src\pipeline.py --help
.\.venv\Scripts\python.exe evaluation\evaluate.py --help
```

## Model 3 isolated environment

IndicTrans2 was tested in a separate Python 3.12 CPU environment. The toolkit recommends modern NumPy/PyTorch/Transformers versions, and its package documentation states Windows is not a tested platform. The project installed and initialized `IndicTransToolkit==1.1.1` on this Windows machine. For a new environment:

```powershell
py -3.12 -m venv .venv_model3_indictrans2
.\.venv_model3_indictrans2\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
python -m pip install -r requirements-model3-indictrans2.txt
```

The requirements file intentionally leaves CPU PyTorch installation explicit. Do not install Flash Attention, CUDA, `torchvision`, or `torchaudio` for this CPU workflow. IndicTrans2's Hugging Face model repository is gated: the user must accept its access terms and authenticate locally before `from_pretrained` can download files. Do not place tokens in project files.

Example pilot invocation (writes a new JSON result path):

```powershell
.\.venv_model3_indictrans2\Scripts\python.exe scripts\test_model3_indictrans2.py `
  --input outputs\pipeline_final\sample-1_final_output.txt `
  --count 5 --group-size 5 --device cpu --num-beams 5 `
  --output-json outputs\translation\indictrans2\sample-1_rerun.json
```

The script expects one sentence per nonempty input line. It supports `--input`, `--model-id`, `--count`, `--group-size`, `--device cpu`, `--num-beams`, `--output-json`, and optional `--output-text`; verify options with `--help`. The tested candidate has been rejected for translation quality; generated outputs are experimental and must remain UNVALIDATED.

## Dependency files

- `requirements.txt`: shared core Model 1/2 dependencies.
- `requirements-model3.txt`: preserved dependency snapshot for the earlier M2M100/CTranslate2 experiment; not required for the selected core pipeline.
- `requirements-model3-indictrans2.txt`: toolkit and compatible Transformers/NumPy/SentencePiece pins for the IndicTrans2 experiment; CPU PyTorch is installed separately as shown above.

Keep these environments isolated because the translation experiments use different Transformers/PyTorch versions.

## Regression tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s evaluation -p "test_*.py" -v
```

The suite last passed 65/65 in the existing project `venv`. Its status for the current documentation pass is recorded in [Evaluation](EVALUATION.md).
