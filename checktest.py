import torch

checkpoint = '/home/urp6/workspace/a-PyTorch-Tutorial-to-Object-Detection/checkpoint_ssd300.pth.tar'



checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
optimizer = checkpoint['optimizer']