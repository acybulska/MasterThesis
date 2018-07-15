import keras
import matplotlib as plt
import numpy as np

model = keras.models.load_model("/Users/lisa/Desktop/ThesisResults/full14.model")
met5 = keras.metrics.top_k_categorical_accuracy(k=3)
print(str(met5))
# plt.style.use("ggplot")
# plt.figure()
# N = 100
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("metrics1.jpg")