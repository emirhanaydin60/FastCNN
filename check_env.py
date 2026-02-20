import sys
print('executable:', sys.executable)
print('version:', sys.version)
try:
    import torch
    print('torch:', torch.__version__)
    from model import FastCNN
    m = FastCNN()
    x = torch.randn(1,1,28,28)
    out = m(x)
    print('forward shape:', tuple(out.shape))
except Exception as e:
    print('error:', repr(e))
