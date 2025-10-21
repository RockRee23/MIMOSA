import torch
def training(model, optimizer, ux_loader, x_type):#x means bundle or item
    model.train()
    
    losses = []
    c_times = []
    for u_idx, x_idx, u_neis, b_neis in ux_loader:
        u_idx = u_idx.squeeze()#u_idx shape:[batch_size, 1]
        model.zero_grad()
        if x_type == 'bnd':
            loss = model.train_bnd_loss(u_idx, x_idx, u_neis, b_neis)
        elif x_type == 'itm':
            loss = model.train_itm_loss(u_idx, x_idx, u_neis, b_neis)
        loss.cuda()
        losses.append(loss)
        loss.backward()
        optimizer.step()
    avg_loss = torch.mean(torch.stack(losses))
    return avg_loss
