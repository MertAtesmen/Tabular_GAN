from sdv.single_table.base import BaseSynthesizer
import os
from datetime import datetime
from pathlib import Path
import secrets
import torch

SAVING_FOLDER = Path('..') / Path('models') 

def save_model(
    model: BaseSynthesizer,
    synthesizer_type: str,
    dataset_name: str,
    model_name: str | None = None,
    folder_path: Path | str = SAVING_FOLDER,
) -> str:
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    synthesizer_folder_path = folder_path / dataset_name / synthesizer_type.upper()
    
    if not synthesizer_folder_path.exists():
        synthesizer_folder_path.mkdir()
        
    if model_name is None:
        random_hash = secrets.token_hex(32)
        model_name = random_hash
        
    file_path = synthesizer_folder_path / f'{model_name}.pth'
    torch.save(model, file_path)
    
    return str(file_path)
    