
## AI8X Installation with UV

If uv is not yet installed in the system <br>
Linux / macOS
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or
```
wget -qO- https://astral.sh/uv/install.sh | sh
```
Windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```
cd ai8x-training
```

```
uv python install 3.11.8
```

```
uv venv --python 3.11.8
```

Linux / macOS
```
source .venv/bin/activate
```
Windows
```
source .venv/Scripts/activate
```

```
uv pip install -U pip wheel setuptools
```

Remove distiller from requirements.txt temporarily
> -e distiller --config-settings editable_mode=strict

CUDA 12
```
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
```

Manually install distiller
```
uv pip install -e distiller --config-settings editable_mode=strict
```

Error

AttributeError: module 'distiller' has no attribute 'knowledge_distillation'

```
uv uninstall distiller
```

```
uv pip install -e distiller --config-settings editable_mode=strict
```