import os
os.environ["RAY_NODE_IP_ADDRESS"] = "127.0.0.1"

from flwr.common import Context
from flwr.client import ClientApp
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Importamos tus módulos
from src.model import get_resnet18_2d
from src.dataset import load_client_data
from src.client import MRINumPyClient

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def _resolve_config_path(run_config):
    """Resuelve el YAML tanto para run_sim.py como para flwr run."""
    raw_path = os.environ.get("FL_CONFIG_PATH") or run_config.get(
        "config_path",
        "configs/experiment_fedavg.yaml",
    )
    path = Path(raw_path)
    candidates = [path if path.is_absolute() else Path.cwd() / path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / "configs" / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No se encuentra el archivo de configuración: {raw_path}")

def client_fn(context: Context):
    """Función que Flower llama para crear un cliente virtual."""

    # 1. Leer el config YAML
    config_path = _resolve_config_path(context.run_config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    # 2. En el Flower moderno, 'cid' pasa a ser 'partition-id' dentro del node_config
    partition_id = context.node_config["partition-id"]
    hospital_id = partition_id + 1 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Cargar el modelo
    model = get_resnet18_2d(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes']
    )
    
    # 4. Cargar datos específicos del hospital
    trainloader = load_client_data(hospital_id, config, split="train")
    testloader = load_client_data(hospital_id, config, split="val")
    
    # 5. Instanciar tu cliente y aplicarle .to_client() para cumplir el estándar nuevo
    numpy_client = MRINumPyClient(model, trainloader, testloader, device, config)
    return numpy_client.to_client()

# Instanciamos la ClientApp moderna
app = ClientApp(client_fn=client_fn)
