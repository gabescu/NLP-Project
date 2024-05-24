from transformers import AutoModelForTokenClassification, trainer, Trainer
import os
from Trainer_code import test_dataset
import numpy as np

cwd = os.getcwd()
saved_path = f"{cwd[0:-11]}Harry_Potter_Model_2"

loaded_model = AutoModelForTokenClassification.from_pretrained(saved_path)
trainer = Trainer(model = loaded_model)

predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis = 2)

with open("predictions and labels.txt", "w") as file:
  file.write(predictions)
  file.write(labels)
