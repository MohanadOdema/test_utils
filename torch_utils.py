import sys

###################################################################################################
### TO use the getActivaiton fn in the main script
### First define two dictionaries for the input and output 
input_activations = {}
output_activations = {}
### Then register the hooks for the layers in questions
### e.g., h_stem_conv3 = model.stem.convs[9].register_forward_hook(getActivation('Conv3'))
### Lastly, remove the hooks 
### e.g., h_stem_conv3.remove()
####################################################################################################


def getActivation(name):
    '''hook fn to retrieve intermeidate activations or their shape of a layer in a model'''
    # the hook signature
    def hook(model, input, output):
        input_activations[name] = input[0].shape
        output_activations[name] = output.shape
    return hook

def conv_as_gemm(ip_mtx, weight, stride=1, padding=1):
    '''Args:
    - input of dims (B, C, n, n)   # B=1
    - weight of dims (K, C, m, m)
    '''
    B, C, n, _ = ip_mtx.shape
    K, _, m, _ = weight.shape
    
    # unroll input
    ip_vector = ip_mtx.view(-1, 1)
    op_dim = int(((n-m+2*padding)/stride)+1)

    # weight broadcast
    n_1, n_2 = op_dim**2, C*n*n
    sparsified_weight = torch.randn(K, n_1, n_2)     # FIXME: retain the same values of original weights
    weight_sparsity_ratio = 1 - (m/n)

    # GEMM (torch.matmul)
    op_vector = torch.matmul(sparsified_weight, ip_vector)
    output = op_vector.view(B, K, op_dim, op_dim)

    import pdb
    pdb.set_trace()

    assert verify_conv_dims(ip_mtx, weight, stride, padding, output.shape)

def fc_as_gemm(ip_vector, op_size):
    N, n = ip_vector.shape
    fc_weight = torch.randn(op_size, n)
    output = torch.matmul(fc_weight, ip_vector.transpose(0,1))
    output = output.view(1, -1)

    assert verify_fc_dims(ip_vector, op_size, output.shape)

def verify_conv_dims(ip_mtx, weight, stride, padding, GEMM_shape):
    K, C, m, _ = weight.shape
    from torch.nn import Conv2d
    conv_out = Conv2d(C, K, m, stride=stride, padding=padding)(ip_mtx)
    print(GEMM_shape, conv_out.shape)
    return GEMM_shape == conv_out.shape

def verify_fc_dims(ip_vector, op_size, GEMM_shape):
    N, n = ip_vector.shape
    from torch.nn import Linear
    linear_out = Linear(n, op_size)(ip_vector)
    print(GEMM_shape, linear_out.shape)
    return GEMM_shape == linear_out.shape

def test_conv_result(input, weight, out):
    from torch.nn import Conv2d
    pass

if __name__ == '__main__':

    ip_mtx_shape = (1,32,8,8)
    weight_mtx_shape = (32,16,4,4)
    stride = 2 
    padding = 1

    import torch

    ip_mtx = torch.randn(ip_mtx_shape)
    weight = torch.randn(weight_mtx_shape)
    # conv_as_gemm(ip_mtx, weight, stride, padding)

    # ip_vector from the matrix
    op_size = 100
    ip_vector = ip_mtx.view(1, -1)
    fc_as_gemm(ip_vector, op_size)