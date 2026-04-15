
v4 dataset fix changes:
- Notebook00 now copies the rebuilt dataset back to /content/drive/MyDrive/EcoCAR/datasets/bdd100k_vehicle5
  and also refreshes /content/drive/MyDrive/EcoCAR/datasets/bdd100k_vehicle5.tar.
- This makes the outputs persistent so notebook01/02/03/06 can reuse them in fresh runtimes.
