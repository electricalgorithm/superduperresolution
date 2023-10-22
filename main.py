"""
This file creates the model, and trains it.
"""
import logging
from sdr_model import Trainer, SuperDuperResolution, MSELoss, optim

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%d.%m.%Y %H:%M",
    filename="training_logs.log",
    filemode="w",
)

# Create a logger.
logger = logging.getLogger(__name__)

# Clear the PIL logger.
logging.getLogger("PIL").setLevel(logging.INFO)

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
logger.info("Hyperparameters are set.")
logger.info("- TRAIN_BATCH_SIZE: %d", TRAIN_BATCH_SIZE)
logger.info("- TEST_BATCH_SIZE: %d", TEST_BATCH_SIZE)
logger.info("- EPOCHS: %d", EPOCHS)
logger.info("- LEARNING_RATE: %f", LEARNING_RATE)


# Create the model.
model = SuperDuperResolution()
logger.info("Model is created.")

# Create the trainer.
trainer = Trainer(
    model=model,
    criterion=MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE),
    train_batch_size=TRAIN_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE,
)
logger.info("Trainer is created.")

trainer.add_videos(".videos")
logger.info("Videos are added to the trainer.")

trainer.train_and_test(
    epochs=EPOCHS, input_dims="640x360", output_dims="854x480", amount_of_data=1000
)
logger.info("Training is completed.")
