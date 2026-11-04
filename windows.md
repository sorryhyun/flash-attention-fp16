# Request: a Windows wheel of the current build

Wanted: one Windows wheel with all five patches, on torch's stable ABI, uploaded to the latest release
next to the Linux wheel:

```
release  v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.12plus
wheel    flash_attn-2.8.3+sm120.hdim64128.fp16acc.fwdpv.cu132.torch2.12plus-cp310-abi3-win_amd64.whl
```

Windows users of `anima_lora` are still on the old per-torch wheel
(`v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14`): the fp16acc backward without the PV forward,
loading on torch 2.14 only. The stable-ABI wheel would load on every torch from 2.12 up, as
the Linux one does.

## What has and has not been compiled with MSVC

- The old Windows wheel was built with `sm120`, `fp16acc`, `hdim64` and `windows.patch`. Those four and
  the flags below are known to work.
- `fwd_fp16acc.patch` and `stable_abi.patch` have only been compiled with gcc. The
  stable ABI is the likely trouble: the extension now reaches libtorch only through
  `torch/csrc/stable`, and the cxx side gets `-DPy_LIMITED_API=0x030a0000`, which links `python3.lib`.
- All six patches apply cleanly, in the order below, on the 2.8.3 sdist (checked on Linux).

The build needs no GPU. Checking the wheel needs an sm_120 card (RTX 50-series).

## Install first

- Visual Studio 2022 BuildTools with the C++ x64 tools and a Windows 11 SDK:

  ```powershell
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
  ```

- CUDA Toolkit 13.2 (`anima_lora`'s `install.ps1` installs it too).
- Git for Windows (for `patch.exe`), [uv](https://docs.astral.sh/uv/) and `gh`.
- RAM: each `cicc` takes about 5 GB, so set `MAX_JOBS` to about RAM in GB / 10. The full build takes
  about 11 minutes at `MAX_JOBS=6` on Linux.

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
foreach ($p in 'sm120', 'fp16acc', 'hdim64', 'fwd_fp16acc', 'stable_abi', 'windows') {
    & 'C:\Program Files\Git\usr\bin\patch.exe' -p1 -i "..\repo\patches\$p.patch"
    if ($LASTEXITCODE) { throw "patch $p" }
}

$env:PATH = "C:\fa2\venv\Scripts;$env:CUDA_PATH\bin;$env:PATH"
$env:CUDA_HOME = $env:CUDA_PATH
$env:DISTUTILS_USE_SDK = '1'
$env:BUILD_TARGET = 'cuda'
$env:FLASH_ATTENTION_FORCE_BUILD = 'TRUE'
$env:FLASH_ATTN_CUDA_ARCHS = '120'
$env:FLASH_ATTN_TORCH_TARGET = '2.12'
$env:FLASH_ATTN_LOCAL_VERSION = 'sm120.hdim64128.fp16acc.fwdpv.cu132.torch2.12plus'
$env:MAX_JOBS = '6'
$env:NVCC_FLAGS = '-w --disable-warnings'
$env:CXXFLAGS = '/w /Zc:preprocessor'
$env:CFLAGS = '/w /Zc:preprocessor'
$env:NVCC_APPEND_FLAGS = '-Xcompiler="/Zc:preprocessor"'
$env:TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES = '1'

python setup.py bdist_wheel --dist-dir=dist
Get-Item dist\*.whl
```

Building against torch 2.14.1 is fine: `FLASH_ATTN_TORCH_TARGET` sets the oldest torch the wheel
loads on, not the one it was built with. The file name must match the one at the top; `anima_lora`
pins it by URL.

| Error | Cause |
|---|---|
| `Hunk FAILED` | The patches came out CRLF; clone with `core.autocrlf=false` |
| `C1189` from `cuda/std/__cccl/preprocessor.h` | `/Zc:preprocessor` did not reach `cl` |
| `C2666` on `ArrayRef`, `C7555`, `C7582` | `windows.patch` is missing, or `DISTUTILS_USE_SDK` is not `1` |
| `data member initializer is not allowed` in `AutogradState.h` | nvcc is not on C++20: `sm120.patch` did not apply |
| A `cicc` dies with no message | Out of memory: lower `MAX_JOBS` |
| Anything in `torch/csrc/stable` or `torch/headeronly`, or an unresolved `aoti_torch_*` at link | New with `stable_abi.patch`. Send the first error and the command line above it |
| Unresolved `Py*` symbols, or `python313.lib` requested | The limited-API build is linking the full CPython ABI. Send the link line |

## Check it on the card

Unpack the wheel and run `smoke.py`. It takes a few seconds.

```powershell
mkdir C:\fa2\unpacked
tar -xf (Get-Item dist\*.whl).FullName -C C:\fa2\unpacked
python ..\repo\smoke.py C:\fa2\unpacked
```

The Linux wheel of the same release prints this on an RTX 5070 Ti. Windows should end in `ok` with
distances within a few ten-thousandths:

```
head dim 64: finite
head dim 128: finite, fp16acc is 0.0015 from stock
forward pv (default): 0.0025 from bf16
forward all: 0.0027 from bf16
head dim 96 raises: this FA2 build holds head dims 64 and 128 only
ok
```

If you have `anima_lora` on the same machine, also run about 20 training steps on it. In its venv:

```powershell
uv pip install --no-deps --force-reinstall (Get-Item C:\fa2\flash_attn-2.8.3\dist\*.whl).FullName
```

Then train MOSS-Image (`max_steps = 20`) and an Anima LoRA for about 20 steps each. The loss should
stay finite, and with `FLASH_ATTN_FP16ACC=0` set the step should get about 5% slower. `uv sync` puts
the pinned wheel back afterwards.

## Sending it back

With write access to the repo, upload after the smoke check passes:

```powershell
gh release upload v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.12plus (Get-Item dist\*.whl).FullName -R sorryhyun/flash-attention-fp16
```

Without it, open an issue with the wheel attached, or a link to it, and include:

- the smoke output, the card and the driver version;
- the step times with and without `FLASH_ATTN_FP16ACC=0`, if you ran `anima_lora`;
- for a failed build, the first error with about 30 lines before it.

After the upload, `anima_lora` moves `WINDOWS_TAG` in `library/runtime/fa2.py` and the `win32` URL
of its `fa2-sm120` group to this release.
