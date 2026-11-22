#!/usr/bin/env bash
set -e

# Remove o modelo que sobrou de execuções anteriores no Colab.
echo 'from pathlib import Path; Path("/content/text_model.keras").unlink(missing_ok=True)' | colab exec

if [ -f text_model.keras ]; then
    colab upload text_model.keras content/text_model.keras
fi

colab exec -f main.py --timeout 0
colab download content/text_model.keras ./text_model.keras