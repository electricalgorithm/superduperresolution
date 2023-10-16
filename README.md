## Super Duper Resolution with CNN

This is a PyTorch implementation of the paper "Image Super-Resolution Using Deep Convolutional Networks" by Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang.

### Usage
```python
from sdr_model import Trainer, SuperDuperResolution, MSELoss, optim

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

trainer = Trainer(
    model=SuperDuperResolution(),
    criterion=MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=LEARNING_RATE),
    train_batch_size=TRAIN_BATCH_SIZE,
    test_batch_size=TEST_BATCH_SIZE,
)

trainer.add_videos("location/to/videos")
trainer.train_and_test(epochs=EPOCHS)
```
