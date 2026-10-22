from utils.dataset import get_datasets

for name in ["faces", "ela"]:
    print(f"--{name}--")
    train, val, test = get_datasets(f"data/samples/{name}", batch_size=4)
    for images, labels in train.take(1):
        print(images.shape, images.dtype, labels.numpy())
        