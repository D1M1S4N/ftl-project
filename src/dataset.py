import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class MSLesSegDataset(Dataset):
    def __init__(self, dataframe, base_path, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame con la columna 'image_path'.
            base_path (str/Path): Directorio raíz del proyecto.
            transform (callable, optional): Transformaciones de PyTorch.
        """
        # Reseteamos el índice por si el DataFrame viene de un train_test_split
        self.dataframe = dataframe.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extraemos la ruta relativa (ej: MRIms_kde/imagesTr/axial/MRIms_XXX_YYY.png)
        img_rel_path = self.dataframe.iloc[idx]['image_path']
        img_full_path = self.base_path / img_rel_path

        # 1. Cargamos la imagen original en escala de grises
        image = Image.open(img_full_path).convert('L')

        # 2. La tarea del proyecto es EM vs control a nivel de paciente/slice.
        # No usamos la máscara para convertir el problema en lesión visible vs sano.
        label = int(self.dataframe.iloc[idx]['label_id'])

        # 3. Aplicamos transformaciones
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def _build_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def _validate_label_source(config):
    label_source = config['data'].get('label_source', 'csv_label_id')
    if label_source != 'csv_label_id':
        raise ValueError(
            "Este proyecto define la tarea como EM vs control; "
            "usa data.label_source='csv_label_id'."
        )

def _split_dataframe_by_patient(df, seed, validation_split):
    """Separa train/val por paciente para evitar fuga entre slices del mismo sujeto."""
    patient_label_counts = df.groupby("patient_id")["label_id"].nunique()
    if (patient_label_counts > 1).any():
        bad_patients = patient_label_counts[patient_label_counts > 1].index.tolist()
        raise ValueError(f"Pacientes con etiquetas inconsistentes: {bad_patients[:5]}")

    patient_df = df[["patient_id", "label_id"]].drop_duplicates().reset_index(drop=True)
    stratify = patient_df["label_id"] if patient_df["label_id"].nunique() > 1 else None

    train_patients, val_patients = train_test_split(
        patient_df,
        test_size=validation_split,
        random_state=seed,
        stratify=stratify,
    )

    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])
    train_df = df[df["patient_id"].isin(train_ids)]
    val_df = df[df["patient_id"].isin(val_ids)]

    return train_df, val_df

def load_client_data(client_id, config, split="train"):
    """
    Carga los datos específicos de un cliente leyendo su CSV correspondiente.
    Realiza un split estratificado por paciente para obtener entrenamiento y validación local.
    """
    _validate_label_source(config)

    # Cargamos directamente tu hospital_x_train.csv
    csv_path = Path("data") / f"hospital_{client_id}_train.csv"
    df = pd.read_csv(csv_path)

    seed = config.get('seed', 42)
    validation_split = config['data'].get('validation_split', 0.2)
    train_df, val_df = _split_dataframe_by_patient(
        df=df,
        seed=seed,
        validation_split=validation_split,
    )

    client_df = train_df if split == "train" else val_df
    transform = _build_image_transform()

    base_path = config['data'].get('base_path', '.')
    dataset = MSLesSegDataset(dataframe=client_df, base_path=base_path, transform=transform)

    is_train = (split == "train")
    return DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=is_train)

def load_global_test_data(config, return_dataframe=False):
    """
    Carga el conjunto de test centralizado (IID) usando global_test.csv.
    """
    _validate_label_source(config)

    test_file = config['data'].get('global_test_file', 'data/global_test.csv')
    df_test = pd.read_csv(Path(test_file))

    transform = _build_image_transform()
    base_path = config['data'].get('base_path', '.')
    dataset = MSLesSegDataset(dataframe=df_test, base_path=base_path, transform=transform)

    loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False)
    if return_dataframe:
        return loader, df_test.reset_index(drop=True)
    return loader
