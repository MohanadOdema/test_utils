### TO use the getActivaiton fn in the main script
### First define two dictionaries for the input and output 
input_activations = {}
output_activations = {}
### Then register the hooks for the layers in questions
### e.g., h_stem_conv3 = model.stem.convs[9].register_forward_hook(getActivation('Conv3'))
### Lastly, remove the hooks 
### e.g., h_stem_conv3.remove()


def getActivation(name):
    '''hook fn to retrieve intermeidate activations or their shape of a layer in a model'''
    # the hook signature
    def hook(model, input, output):
        input_activations[name] = input[0].shape
        output_activations[name] = output.shape
    return hook