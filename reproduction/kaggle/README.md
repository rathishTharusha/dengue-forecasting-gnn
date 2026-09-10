# Kaggle CLI setup

The exact reproductions run on Kaggle because they need a pinned, Linux,
Python-3.11 environment that this Windows/Drive working folder cannot provide —
specifically `torch==2.1.2` with matching `torch-scatter` / `torch-sparse`
wheels, which is what the authors' `requirements.txt` pins.

---

## 1. Install the CLI

Already present on this machine (`kaggle` resolves on PATH). To install elsewhere:

```bash
pip install kaggle
```

## 2. Authenticate — you must do this yourself

**Never paste a token into a chat, an issue, or a file inside this folder.**
This folder syncs to Google Drive.

The installed CLI is **kaggle 2.2.4**, which uses access tokens, not the older
`kaggle.json` username/key pair. It resolves credentials in this order
(`kagglesdk.kaggle_env.get_access_token_from_env`):

1. `KAGGLE_API_TOKEN` environment variable
2. `~/.kaggle/access_token` (or `access_token.txt`)
3. `~/.kaggle/kaggle.json` (legacy)

### Preferred: OAuth, nothing to store

```
kaggle auth login
```

Opens a browser flow and caches credentials for you. No token file to create,
leak, or get the encoding wrong on.

### Alternative: a token file

Generate one at <https://www.kaggle.com/settings/api> → **Generate New Token**,
then, in **cmd.exe** — not PowerShell:

```
if not exist "%USERPROFILE%\.kaggle" mkdir "%USERPROFILE%\.kaggle"
> "%USERPROFILE%\.kaggle\access_token" echo YOUR_TOKEN_HERE
icacls "%USERPROFILE%\.kaggle\access_token" /inheritance:r /grant:r "%USERNAME%:F"
```

> **Why cmd.exe and not PowerShell.** PowerShell 5.1's `>` redirect and
> `Out-File` write **UTF-16LE with a BOM**. The Kaggle SDK reads the file with a
> plain `Path.read_text()`, so a BOM makes the token unreadable — a 37-character
> token lands as an 80-byte file and authentication fails with a confusing
> error. `cmd.exe`'s `echo` writes plain bytes. Notepad adds a UTF-8 BOM, so
> avoid that too.
>
> Putting the redirect *before* `echo` avoids the trailing space cmd.exe would
> otherwise include. The SDK strips whitespace, so it is belt-and-braces.
>
> If you must use PowerShell:
> ```powershell
> [IO.File]::WriteAllText("$env:USERPROFILE\.kaggle\access_token", "YOUR_TOKEN_HERE")
> ```

`setup_kaggle.py --check` detects a BOM and tells you exactly how to fix it.

## 3. Verify

```bash
python reproduction/kaggle/setup_kaggle.py --check
```

Confirms the CLI is installed, a credential exists, its encoding is usable, and
the API authenticates. It never decodes, prints or transmits the token — only
its leading bytes and length.

## 4. Push a reproduction kernel

Each kernel lives in `kernels/<name>/` with its notebook and a
`kernel-metadata.json`. The metadata needs your Kaggle username, which the setup
script fills in:

```bash
python reproduction/kaggle/setup_kaggle.py --init-metadata
```

An access token is opaque, so unlike the legacy `kaggle.json` there is no
username inside it to read. The script asks the API instead; if that fails, pass
it yourself:

```bash
python reproduction/kaggle/setup_kaggle.py --init-metadata --username <your-username>
```

Then, per kernel:

```bash
cd reproduction/kaggle/kernels/weng-2024-exact-reproduction-stgat
kaggle kernels push
kaggle kernels status <your-username>/weng-2024-exact-reproduction-stgat
kaggle kernels output <your-username>/weng-2024-exact-reproduction-stgat -p ../../../results/
```

`kaggle kernels push` uploads and queues a run. `status` polls it. `output`
downloads the log and any files the notebook wrote.

---

## Why the slugs are verbose

Kaggle requires a kernel's `id` slug to match the slug its `title` resolves to.
A mismatch is answered with:

```
400 Client Error: Bad Request
Your kernel title does not resolve to the specified id.
```

— or, worse, the push silently succeeds under the title's slug instead of the
one you asked for. `gen_kernels.py` therefore derives the directory name, the
notebook filename and the `id` from `slugify(title)`, so the three cannot drift.
Titles use only letters, digits and spaces to keep that transformation
predictable.

## Kernel settings that matter for reproduction

Set in each `kernel-metadata.json`, and each one is a reproduction requirement,
not a preference:

| Setting | Value | Why |
|---|---|---|
| `enable_internet` | `true` | The notebook clones the authors' repo and installs pinned wheels |
| `enable_gpu` | `false` | The paper states the models were trained on a CPU (§II.D: "AMD Ryzen 5 PRO 4650U CPU") |
| `is_private` | `true` | Nothing here is ours to publish |

**One model per kernel.** Kaggle caps a session at 12 hours. Running all five
architectures in one kernel risks hitting that cap and losing the whole run, so
each is pushed separately. Each kernel calls the authors' own `run_*` function
with the authors' own segment list, which is exactly what their
`if __name__ == "__main__"` block does.

---

## Cost and time

Kaggle CPU notebooks are free with a weekly quota (30 h/week at time of writing;
check your account). Expect roughly 1–4 hours per architecture for 5 segments ×
50 epochs. Push them one at a time and check `status` rather than queuing all
five at once.
