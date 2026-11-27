# metronome in Veda

## Analyze the Rigveda

metronome: https://github.com/bnagy/metronome/
text: https://github.com/VedaWebProject/vedaweb-data 

## Installation

`Metronome` is not published on PyPI, so install it from GitHub first, then install the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/bnagy/metronome@main
pip install -r requirements.txt
```

Running `pip install -r requirements.txt` on its own fails because `Metronome==0.2.3` cannot be found on PyPI. Once Metronome is installed from GitHub, that requirement is already satisfied.
