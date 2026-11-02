# Building the Windows wheel

The Windows wheel has the same patches and head dims (64, 128) as the Linux one, built for
CPython 3.13 and torch 2.14.1+cu132, the Windows pin of `anima_lora`. It goes in the same release:

```
flash_attn-2.8.3+sm120.hdim64128.fp16acc.cu132torch2.14-cp313-cp313-win_amd64.whl
```

`anima_lora` looks for exactly that name, so keep `FLASH_ATTN_LOCAL_VERSION` as below.

No Windows wheel has been built yet. Nothing here has run on Windows, and the kernels have not been
compiled with MSVC. The flags come from
[mjun0812's `build_windows.ps1`](https://github.com/mjun0812/flash-attention-prebuild-wheels/blob/main/build_windows.ps1),
which builds the pinned wheels.

## What differs from Linux

- `patches/windows.patch`: `sm120.patch` moves nvcc to C++20, but the host side of FA2's Windows branch
  stays on `/std:c++17`. torch 2.14's headers need `/permissive- /std:c++20` there.
- CUDA 13.2's CCCL headers raise `C1189` under MSVC's traditional preprocessor, so `cl` needs
  `/Zc:preprocessor`, both directly and through nvcc.
- `DISTUTILS_USE_SDK=1` is what makes `setup.py` take the Windows branch at all.

## Install first

- Visual Studio 2022 BuildTools with the C++ x64 tools and a Windows 11 SDK:

  ```powershell
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
  ```

- CUDA Toolkit 13.2. `anima_lora`'s `install.ps1` installs it too.
- Git for Windows (for `patch.exe`), [uv](https://docs.astral.sh/uv/) and `gh`.

The build needs no GPU. Each `cicc` takes about 5 GB, so set `MAX_JOBS` to about RAM in GB / 10.
Checking the wheel needs an sm_120 card.

## Build

In PowerShell. Keep the folder short and clone with `core.autocrlf=false`: a CRLF patch does not apply
to the LF sources.

```powershell
Import-Module 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
Enter-VsDevShell -VsInstallPath 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' -DevCmdArguments '-arch=x64 -host_arch=x64' -SkipAutomaticLocation

mkdir C:\fa2; cd C:\fa2
git -c core.autocrlf=false clone https://github.com/sorryhyun/flash-attention-fp16 repo

curl.exe -sSLO https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
if ((Get-FileHash flash_attn-2.8.3.tar.gz -Algorithm SHA256).Hash -ne '1E71DD64A9E0280E0447B8A0C2541BAD4BF6AC65BDEAA2F90E51A9E57DE0370D') { throw 'sdist hash' }
tar -xzf flash_attn-2.8.3.tar.gz

uv venv -p 3.13 venv
uv pip install -p venv setuptools==75.8.0 wheel packaging psutil numpy ninja
uv pip install -p venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132

cd flash_attn-2.8.3
foreach ($p in 'sm120', 'fp16acc', 'hdim64', 'windows') {
    & 'C:\Program Files\Git\usr\bin\patch.exe' -p1 -i "..\repo\patches\$p.patch"
    if ($LASTEXITCODE) { throw "patch $p" }
}

$env:PATH = "C:\fa2\venv\Scripts;$env:CUDA_PATH\bin;$env:PATH"
$env:CUDA_HOME = $env:CUDA_PATH
$env:DISTUTILS_USE_SDK = '1'
$env:BUILD_TARGET = 'cuda'
$env:FLASH_ATTENTION_FORCE_BUILD = 'TRUE'
$env:FLASH_ATTN_CUDA_ARCHS = '120'
$env:FLASH_ATTN_LOCAL_VERSION = 'sm120.hdim64128.fp16acc.cu132torch2.14'
$env:MAX_JOBS = '6'
$env:NVCC_FLAGS = '-w --disable-warnings'
$env:CXXFLAGS = '/w /Zc:preprocessor'
$env:CFLAGS = '/w /Zc:preprocessor'
$env:NVCC_APPEND_FLAGS = '-Xcompiler="/Zc:preprocessor"'
$env:TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES = '1'

python setup.py bdist_wheel --dist-dir=dist
Get-Item dist\*.whl
```

| Error | Cause |
|---|---|
| `Hunk FAILED` | The patches came out CRLF; clone with `core.autocrlf=false` |
| `C1189` from `cuda/std/__cccl/preprocessor.h` | `/Zc:preprocessor` did not reach `cl` |
| `C2666` on `ArrayRef`, `C7555`, `C7582` | `windows.patch` is missing, or `DISTUTILS_USE_SDK` is not `1` |
| `data member initializer is not allowed` in `AutogradState.h` | nvcc is not on C++20: `sm120.patch` did not apply |
| A `cicc` dies with no message | Out of memory: lower `MAX_JOBS` |

## Check it on the card

Unpack the wheel, then run `smoke.py`. It takes a few seconds.

```powershell
mkdir C:\fa2\unpacked
tar -xf (Get-Item dist\*.whl).FullName -C C:\fa2\unpacked
python ..\repo\smoke.py C:\fa2\unpacked
```

Expected, ending in `ok`, with the distance around 0.002:

```
head dim 64: finite
head dim 128: finite, fp16acc is 0.0015 from stock
head dim 96 raises: this FA2 build holds head dims 64 and 128 only
ok
```

Then run real steps in `anima_lora`. Unpack the same wheel to
`models\fa2_sm120\v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14` and train MOSS-Image and Anima for
about 20 steps each (MOSS: `max_steps = 20`). For each:

- The log says `FlashAttention: sm_120 build ...`. `FlashAttention: pinned wheel (...)` means it did not
  load, and says why.
- The loss stays finite.
- With `FLASH_ATTN_FP16ACC=0` the step gets about 5% slower. On Linux the same change goes from 1022 ms
  to 1071–1082 ms on MOSS, and from 404.7 ms to 426.5 ms on Anima.

## Upload

After the checks pass:

```powershell
gh release upload v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14 (Get-Item dist\*.whl).FullName -R sorryhyun/flash-attention-fp16
```

Once the repo is public, `anima_lora` fetches it with `python tasks.py fa2-sm120`, and its install scripts
do the same on an sm_120 card.
