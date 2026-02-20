import torch
from model import FastCNN
from torchvision import datasets, transforms


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        img, label = ds[0]
        x = img.unsqueeze(0)
        source = 'MNIST test sample'
    except Exception:
        x = torch.randn(1, 1, 28, 28)
        label = None
        source = 'random'

    model = FastCNN()
    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = out.squeeze(0)
        pred = int(torch.argmax(probs).item())

    print('source:', source)
    print('gt_label:', label)
    print('pred:', pred)
    print('probs (first 10):', [float(p) for p in probs.tolist()])


if __name__ == '__main__':
    main()
