import Harry_Potter_Model_2
from Trainer_code import test_dataset

predictions, labels, _ = Harry_Potter_Model_2.predict(test_dataset)

with open("predictions_and_labels.txt", "w") as file:
  file.write(predictions)
  file.write(f"\n{labels}")
