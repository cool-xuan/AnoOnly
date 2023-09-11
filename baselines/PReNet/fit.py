import torch
from torch.autograd import Variable
from baseline.PReNet.utils import sampler_pairs

def fit(X_train_tensor, y_train, model, optimizer, epochs, batch_num, batch_size,
         s_a_a, s_a_u, s_u_u, device=None, anomaly_only=False, scheduler=None):
    # epochs
    for epoch in range(epochs):
        # generate the batch samples
        X_train_loader, y_train_loader = sampler_pairs(X_train_tensor, y_train, epoch, batch_num, batch_size,
                                                       s_a_a=s_a_a, s_a_u=s_a_u, s_u_u=s_u_u)
        for i in range(len(X_train_loader)):
            X_left, X_right = X_train_loader[i][0], X_train_loader[i][1]
            y = y_train_loader[i]

            #to device
            X_left = X_left.to(device); X_right = X_right.to(device); y = y.to(device)
            # to variable
            X_left = Variable(X_left); X_right = Variable(X_right); y = Variable(y)

            # clear gradient
            model.zero_grad()

            # loss forward
            if anomaly_only:
                score = model(X_left, X_right)
                score = score[y!=s_u_u]
                y = y[y!=s_u_u]
                loss = torch.mean(torch.abs(y - score))
            else:
                score = model(X_left, X_right)
                loss = torch.mean(torch.abs(y - score))

            # loss backward
            loss.backward()
            # update model parameters
            optimizer.step()
        if scheduler:
            scheduler.step()
            
def fit_sam(X_train_tensor, y_train, model, optimizer, scheduler, epochs, batch_num, batch_size,
            s_a_a, s_a_u, s_u_u, device=None):
    # epochs
    for epoch in range(epochs):
        # generate the batch samples
        X_train_loader, y_train_loader = sampler_pairs(X_train_tensor, y_train, epoch, batch_num, batch_size,
                                                       s_a_a=s_a_a, s_a_u=s_a_u, s_u_u=s_u_u)
        for i in range(len(X_train_loader)):
            X_left, X_right = X_train_loader[i][0], X_train_loader[i][1]
            y = y_train_loader[i]

            #to device
            X_left = X_left.to(device); X_right = X_right.to(device); y = y.to(device)
            
            # first forward
            score = model(X_left, X_right)
            loss = torch.mean(torch.abs(y - score))
            loss.backward()
            optimizer.ascent_step()
            
            # second forward
            score = model(X_left, X_right)
            loss = torch.mean(torch.abs(y - score))
            loss.backward()
            optimizer.descent_step()
            
        if scheduler:
            scheduler.step()
            
def fit_mysam(X_train_tensor, y_train, model, optimizer, scheduler, epochs, batch_num, batch_size,
              s_a_a, s_a_u, s_u_u, device=None):
    # epochs
    for epoch in range(epochs):
        # generate the batch samples
        X_train_loader, y_train_loader = sampler_pairs(X_train_tensor, y_train, epoch, batch_num, batch_size,
                                                       s_a_a=s_a_a, s_a_u=s_a_u, s_u_u=s_u_u)
        for i in range(len(X_train_loader)):
            X_left, X_right = X_train_loader[i][0], X_train_loader[i][1]
            y = y_train_loader[i]

            #to device
            X_left = X_left.to(device); X_right = X_right.to(device); y = y.to(device)
            
            # first forward
            score = model(X_left, X_right)
            loss = torch.mean(torch.abs(y - score))
            loss.backward()
            optimizer.ascent_step()
            
            # second forward
            score = model(X_left, X_right)
            losses = torch.abs(y - score)
            losses[y == s_a_u] = losses[y == s_a_u] * 0.5
            losses[y == s_u_u] = losses[y == s_u_u] * 0.0
            loss = torch.mean(losses)
            loss.backward()
            optimizer.descent_step()
            
        if scheduler:
            scheduler.step()
            

def fit_nor_base_ano_sam(X_train_tensor, y_train, model, optimizer, scheduler, epochs, batch_num, batch_size,
              s_a_a, s_a_u, s_u_u, device=None):
    # epochs
    for epoch in range(epochs):
        # generate the batch samples
        X_train_loader, y_train_loader = sampler_pairs(X_train_tensor, y_train, epoch, batch_num, batch_size,
                                                       s_a_a=s_a_a, s_a_u=s_a_u, s_u_u=s_u_u)
        for i in range(len(X_train_loader)):
            X_left, X_right = X_train_loader[i][0], X_train_loader[i][1]
            y = y_train_loader[i]

            #to device
            X_left = X_left.to(device); X_right = X_right.to(device); y = y.to(device)
            
            aa_index = y == s_a_a
            au_index = y == s_a_u
            uu_index = y == s_u_u
            
            # normals
            score   = model(X_left[uu_index], X_right[uu_index])
            loss_uu = torch.mean(torch.abs(y[uu_index] - score))
            
            score   = model(X_right[au_index], X_left[au_index], detach_right=True) # disable training on anomaly samples
            loss_au = torch.mean(torch.abs(y[au_index] - score))
            
            loss = loss_uu + loss_au
            loss.backward()
            optimizer.optimizer.step()
            optimizer.optimizer.zero_grad()
            
            # anomaly 
            # first forward
            score   = model(X_left[au_index], X_right[au_index], detach_right=True) # disable training on normal samples
            loss_au = torch.mean(torch.abs(y[au_index] - score))
            
            score   = model(X_left[aa_index], X_right[aa_index])
            loss_aa = torch.mean(torch.abs(y[aa_index] - score))
            
            loss = loss_aa + loss_au
            loss.backward()
            optimizer.ascent_step()
            
            # second forward
            score   = model(X_left[au_index], X_right[au_index], detach_right=True) # disable training on normal samples
            loss_au = torch.mean(torch.abs(y[au_index] - score))
            
            score   = model(X_left[aa_index], X_right[aa_index])
            loss_aa = torch.mean(torch.abs(y[aa_index] - score))
            
            loss = loss_aa + loss_au
            loss.backward()
            optimizer.descent_step()
            
        if scheduler:
            scheduler.step()