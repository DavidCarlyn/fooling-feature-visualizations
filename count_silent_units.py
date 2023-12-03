
# https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model = MyModel()
model.fc2.register_forward_hook(get_activation('fc2'))
x = torch.randn(1, 25)
output = model(x)
print(activation['fc2'])