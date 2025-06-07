import csv


class LossLogger:

    def __init__(self, file_path):
            self.file_path = file_path
            with open(self.file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss", "val_loss"])