import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

class SimpleMeanFedProx(fl.server.strategy.FedProx):
    def aggregate_fit(self, server_round, results, failures):
        """Agrega los modelos usando la media simple, ignorando el volumen de datos."""
        
        # Si no hay resultados de los clientes, devolvemos vacío
        if not results:
            return None, {}
        
        # Extraemos los pesos de los modelos que envían los clientes
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        
        # Calculamos la media simple por cada capa de la red
        # zip(*weights_results) agrupa la capa N de todos los clientes juntos
        aggregated_ndarrays = [
            np.mean(layer_weights, axis=0)
            for layer_weights in zip(*weights_results)
        ]
        
        # Devolvemos los nuevos pesos globales empaquetados para Flower
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Podemos agregar métricas personalizadas aquí si es necesario
        metrics_aggregated = {}
        
        return parameters_aggregated, metrics_aggregated