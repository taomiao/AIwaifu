import torch

if __name__=="__main__":

    x = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    loss_fn = torch.nn.L1Loss()
    y = torch.zeros_like(x)+0.1
    loss = loss_fn(x, y)
    loss.backward()
    print(x, y)
    print(loss)
    opt = torch.optim.SGD([x], lr=0.1)
    for i in range(10):
        opt.step()
        print(x)