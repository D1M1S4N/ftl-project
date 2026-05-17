import flwr as fl
import torch
import torch.nn as nn
import time
from collections import OrderedDict
from src.metrics import calculate_classification_metrics

class MRINumPyClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device, config):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.config = config
        
        # Función de pérdida clásica para clasificación binaria
        self.criterion = nn.CrossEntropyLoss()
        
        # El optimizador se inicializa con el learning rate definido en el spec-kit (yaml)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['client_training']['learning_rate']
        )

    def get_parameters(self, config):
        """Extrae los pesos del modelo de PyTorch para enviarlos al servidor de Flower."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Inyecta los pesos globales recibidos del servidor en el modelo local."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Bucle de entrenamiento local."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()

        epochs = self.config['client_training']['local_epochs']
        strategy_name = self.config['federated_learning'].get('strategy', 'FedAvg')
        default_mu = self.config['federated_learning'].get('proximal_mu', 0.0)
        proximal_mu = float(config.get("proximal_mu", default_mu if strategy_name == "FedProx" else 0.0))
        global_parameters = [
            param.detach().clone().to(self.device)
            for param in self.model.parameters()
        ] if proximal_mu > 0.0 else []
        start_time = time.time()

        for epoch in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if proximal_mu > 0.0:
                    proximal_term = torch.tensor(0.0, device=self.device)
                    for local_param, global_param in zip(self.model.parameters(), global_parameters):
                        proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss = loss + (proximal_mu / 2.0) * proximal_term

                loss.backward()
                self.optimizer.step()

                # --- DESTRUCCIÓN EXPLÍCITA DE TENSORES ---
                del images, labels, outputs, loss
                # -----------------------------------------

        end_time = time.time()
        time_per_epoch = (end_time - start_time) / epochs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "time_per_epoch": time_per_epoch,
            "proximal_mu": proximal_mu,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()
        
        loss = 0.0
        all_preds, all_labels = [], []
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # --- DESTRUCCIÓN EXPLÍCITA DE TENSORES ---
                del images, labels, outputs
                # -----------------------------------------

        metrics = calculate_classification_metrics(all_labels, all_preds)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return float(loss / len(self.testloader)), len(self.testloader.dataset), metrics
