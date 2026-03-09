import torch
from torchviz import make_dot
from models.MLP import MLP


input_size   = 784
hidden_sizes = [256, 128]
num_classes  = 10
dropout      = 0
activation   = torch.nn.ReLU
batchnorm    = True


model = MLP(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    num_classes=num_classes,
    dropout=dropout,
    activation=activation,
    batchnorm=batchnorm
)


x = torch.randn(4, 1, 28, 28)  # batch size 1, 1x28x28 MNIST image

y = model(x)


dot = make_dot(y, params=dict(model.named_parameters()))

dot.render("mlp_visualization", format="png")  
print("visualization is done")
