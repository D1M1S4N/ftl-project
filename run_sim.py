import os
import argparse

os.environ["RAY_NODE_IP_ADDRESS"] = "127.0.0.1"
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("FL_CONFIG_PATH", "configs/experiment_fedavg.yaml")


def main():
    parser = argparse.ArgumentParser(description="Lanza la simulacion federada Flower/Ray.")
    parser.add_argument(
        "--config",
        default=os.environ["FL_CONFIG_PATH"],
        help="Ruta al YAML de configuracion del experimento.",
    )
    args = parser.parse_args()
    os.environ["FL_CONFIG_PATH"] = args.config

    print("\n" + "=" * 50)
    print("--INICIANDO SIMULACIÓN FEDERADA--")
    print(f"Config: {os.environ['FL_CONFIG_PATH']}")
    print("=" * 50 + "\n")

    import ray
    import flwr as fl

    from src.server_app import app as server_app
    from src.client_app import app as client_app

    ray.init(ignore_reinit_error=True, _node_ip_address="127.0.0.1")

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=4,
        backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.25}},
    )


if __name__ == "__main__":
    main()
