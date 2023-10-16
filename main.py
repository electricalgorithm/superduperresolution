"""
This file creates the model, and trains it.
"""
from sdr_model import Trainer, SuperDuperResolution, MSELoss, optim

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


# Create the model.
model = SuperDuperResolution()

# Create the trainer.
trainer = Trainer(
    model=model,
    criterion=MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE),
    train_batch_size=TRAIN_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE,
)

trainer.add_videos("videos")
trainer.train_and_test(epochs=EPOCHS)
