# dataset_vecteurs.py

import click
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt

from ptsdae.model import train_standard, predict_standard  # Autoencodeur entraînement et prédiction
from ptsdae.ae import AutoEncoder  # Classe AutoEncoder
from ptdec.dec import DEC  # Classe DEC pour le clustering
from ptdec.model import train, predict


class CSVTextDataset(Dataset):
    def __init__(self, csv_file, cuda=False):
        self.dataframe = pd.read_csv(csv_file)
        self.identifiants = self.dataframe['Index'].tolist()
        self.dataframe['Vector'] = self.dataframe['Vector'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.vectors = self.dataframe['Vector'].tolist()
        self.vectors = torch.tensor(self.vectors, dtype=torch.float32)
        if cuda:
            self.vectors = self.vectors.cuda()

    def __getitem__(self, index):
        vector = self.vectors[index]
        identifiant = self.dataframe.iloc[index, 0]
        return vector, identifiant

    def __len__(self):
        return len(self.vectors)


@click.command()
@click.option(
    "--cuda", help="Utiliser CUDA (par défaut False).", type=bool, default=False
)
@click.option(
    "--batch-size", help="Taille des batches (par défaut 32).", type=int, default=32
)
@click.option(
    "--pretrain-epochs",
    help="Nombre d'époques pour le pré-entraînement (par défaut 100).",
    type=int,
    default=300,
)
@click.option(
    "--finetune-epochs",
    help="Nombre d'époques pour le finetuning (par défaut 200).",
    type=int,
    default=500,
)
@click.option(
    "--hidden-dimension",
    help="Dimension de l'espace latent (par défaut 128).",
    type=int,
    default=128,
)
@click.option(
    "--clusters",
    help="Nombre de clusters pour DEC (par défaut 20).",
    type=int,
    default=10,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, hidden_dimension, clusters):
    writer = SummaryWriter()  # TensorBoard writer

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars("data/autoencoder",
                           {"lr": lr, "loss": loss, "validation_loss": validation_loss},
                           epoch)

    # Charger les datasets
    train_csv = 'DEC/vectors_melange.csv'
    ds_train = CSVTextDataset(csv_file=train_csv, cuda=cuda)
    ds_val = CSVTextDataset(csv_file=train_csv, cuda=cuda)  # Même dataset pour validation

    # Vérifier les données
    print(ds_train[0])  # Affiche le premier élément du dataset

    # Définir les dimensions de l'autoencodeur
    autoencoder_dimensions = [512, 256, hidden_dimension]

    # Initialiser l'autoencodeur
    autoencoder = AutoEncoder(
        input_dimension=autoencoder_dimensions[0],  # 512
        hidden_dimension=autoencoder_dimensions[-1],  # 128
        tied=False
    )

    if cuda:
        autoencoder.cuda()

    # Définir l'optimiseur et la fonction de perte
    optimizer = Adam(autoencoder.parameters(), lr=1e-3)     #0.001
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    # Normaliser les données
    scaler = StandardScaler()
    scaled_train_vectors = scaler.fit_transform(ds_train.vectors.cpu().numpy())
    ds_train.vectors = torch.tensor(scaled_train_vectors, dtype=torch.float32)
    scaled_val_vectors = scaler.transform(ds_val.vectors.cpu().numpy())
    ds_val.vectors = torch.tensor(scaled_val_vectors, dtype=torch.float32)

    # Entraînement de l'autoencodeur
    print("Entraînement de l'AutoEncoder.")
    train_standard(
        dataset=ds_train,
        autoencoder=autoencoder,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        validation=ds_val,
        cuda=cuda,
        silent=False,
        update_freq=1,
        update_callback=training_callback,
        num_workers=0,
        epoch_callback=None,
    )
    print("Entraînement de l'AutoEncoder terminé.")

    # Extraction des encodages
    print("Extraction des vecteurs encodés.")
    encoded_vectors = predict_standard(
        dataset=ds_train,
        model=autoencoder,
        batch_size=32,
        cuda=cuda,
        silent=False,
        encode=True,
    )
    encoded_vectors = encoded_vectors.cpu().numpy()

    # DEC stage (Clustering)
    print("Clustering avec DEC.")
    model = DEC(cluster_number=clusters, hidden_dimension=autoencoder_dimensions[-1], encoder=autoencoder.encoder)
    if cuda:
        model.cuda()

    dec_optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)   #0.01
    start_time = time.time()
    train(
        dataset=ds_train,
        model=model,
        epochs=500,  # Nombre d'époques pour DEC
        batch_size=32,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
    )
    dec_time = time.time() - start_time
    print(f"Clustering avec DEC terminé en {dec_time:.2f} secondes.")

    # Prédiction des clusters
    print("Prédiction des clusters.")
    predicted = predict(
        ds_train, model, 1024, silent=True, return_actual=False, cuda=cuda
    )
    predicted = predicted.cpu().numpy()
    print(predicted)

    # Sauvegarder les clusters dans un fichier CSV
    clusters = {}
    for (vector, identifiant), cluster_id in zip(ds_train, predicted):
        clusters.setdefault(cluster_id, []).append(identifiant)

    output_csv = 'DEC/clusters_nouveaux_10.csv'
    with open(output_csv, 'w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(['Cluster', 'Identifiants'])
        for cluster_id, ids in clusters.items():
            writer_csv.writerow([cluster_id, ', '.join(map(str, ids))])

    print(f"Résultats sauvegardés dans {output_csv}")
    writer.close()


if __name__ == "__main__":
    main()
