import matplotlib.pyplot as plt

def show_images(images) -> None:

    n = images.size(0)
    f = plt.figure(figsize=(24, 6))
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        img = images[i].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(img.clip(0, 1))
        plt.axis('off')
    plt.show(block=True)

def show_images_withPred(images, label, pred) -> None:

    n = images.size(0)
    f = plt.figure(figsize=(24, 6))
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        img = images[i].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(img.clip(0, 1))
        plt.title("{} -> {}".format(label[i], pred[i]))
        plt.axis('off')
    plt.show(block=True)
