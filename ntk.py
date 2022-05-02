import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import os, math, argparse, random, pickle, json, sys

from functorch import make_functional, vmap, vjp, jvp, jacrev
from torchvision import datasets, transforms

from torchsummary import summary

device = 'cuda' if torch.cuda.is_available else 'cpu'



###### Model Init: ######
class CNN(nn.Module):
    def __init__(self,ch1:int=32, ch2:int = 32, ch3:int=32):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, ch1, (3, 3))
        self.conv2 = nn.Conv2d(ch1, ch2, (3, 3))
        self.conv3 = nn.Conv2d(ch2, ch3, (3, 3))
        self.fc = nn.Linear(22*22*ch3, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self,n_in:int,n_h1:int,n_h2:int,n_h3:int,n_out:int):
        super(MLP, self).__init__()
        self.n_in = n_in
        self.fc1 = nn.Linear(n_in, n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.fc3 = nn.Linear(n_h2, n_h3)
        self.out = nn.Linear(n_h3, n_out)
        
    def forward(self, x):
        x = self.fc1(x.reshape(-1,self.n_in))
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        x = x.relu()
        x = self.out(x)
        return x

    
    
###### NTK Helper Methods: ######
def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)

def empirical_ntk(fnet_single, params, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False
        
    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

def empirical_ntk_implicit(func, params, x1, x2, compute='full'):
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)
        
    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_implicit are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    
    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    if compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)
    
    

###### Model Training: ######
def train(model,optimizer,scheduler,train_loader,test_loader,hidden_size:int,num_epochs:int=100,patience:int=5,save_dir:str=None):
    early_stop = 0
    best_acc = 0.
    stats = {'train_loss':{'batch':[],'epoch':[],},'test_acc':[]}
    print('model training:')
    for epoch in range(num_epochs):
        cur_loss = 0.
        for t,(x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(x.to(device))
            loss = F.cross_entropy(logits,y.to(device))
            loss.backward()
            optimizer.step()
            cur_loss += loss.item()
            stats['train_loss']['batch'].append(loss.item())
        
        cur_loss = cur_loss/t
        test_acc = 0.
        n_correct = 0.
        with torch.no_grad():
            n_test = 0.
            n_correct = 0.
            for (x,y) in test_loader:
                pred = model(x.to(device)).max(-1)[1]
                n_correct += (pred == y.to(device)).sum()
                n_test += y.size(0)
            test_acc = n_correct/n_test
        
        print(f'({epoch}) train loss = {cur_loss:.8f} || test acc. = {test_acc:.5f}')
        sys.stdout.flush()
        stats['test_acc'].append(test_acc)
        stats['train_loss']['epoch'].append(cur_loss)
        
        if best_acc < test_acc:
            early_stop = 0
            best_acc = test_acc
            torch.save({"model_dict": model.state_dict()}, os.path.join(save_dir,f"{model.__class__.__name__}_{hidden_size}.pt"))
        else:
            early_stop += 1
            scheduler.step()
            
        if patience <= early_stop:
            print(f'''{'='*8} Eearly stopping at epoch {epoch} {'='*8}''')
            break
            
    return best_acc, stats

    
    
###### Kernel database: we will use subset of the MNIST training set to with N-many imgs per digit and use the rest as test set. ######
def divide_dataset(dataset, N:int):
    # sort the dataset idx according to the labels.
    sorted_labels, sorted_idx = dataset.targets.sort()
    n_img_per_digit = sorted_labels.unique_consecutive(return_counts=True)[1]
    
    select_idx = []
    other_idx = []
    
    idx_offset = 0
    for n_img in n_img_per_digit:
        select_idx.extend(sorted_idx[idx_offset:idx_offset+N].tolist())
        other_idx.extend(sorted_idx[idx_offset+N:idx_offset+n_img].tolist())
        idx_offset += n_img.item()
        
    return select_idx, other_idx



###### NTK inverter ######
def get_H_inv(model, x_train, per_class_n_train:int,block_size:int,hidden_size:int,n_class:int=10,save_dir:str=None):
    assert per_class_n_train % block_size == 0, "per_class_n_train must be divisible by block_size."
    assert per_class_n_train // block_size < 1000, "number of blocks must be less than 1000."
    
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    with torch.no_grad():
        fnet, params = make_functional(model)
        n = (per_class_n_train*n_class) // block_size
        b = block_size
        print(f'computing H by {n} x {n} blocks...')
        for i in range(n):
            for j in range(i, n):
                H = empirical_ntk(fnet_single, params, x_train[i*b:(i+1)*b,...], x_train[j*b:(j+1)*b,...], 'trace')
                torch.save(H, os.path.join(save_dir,f'{str(i).zfill(3)}{str(j).zfill(3)}.pt'))
                del H
                torch.cuda.empty_cache()
        print('loading H by blocks...')
        H = torch.zeros(per_class_n_train*n_class,per_class_n_train*n_class,device=device)
        for i in range(0,n):
            for j in range(i,n):
                H[i*b:(i+1)*b,j*b:(j+1)*b] = torch.load(os.path.join(save_dir,f'{str(i).zfill(3)}{str(j).zfill(3)}.pt'))
                H[j*b:(j+1)*b,i*b:(i+1)*b] = torch.load(os.path.join(save_dir,f'{str(i).zfill(3)}{str(j).zfill(3)}.pt')).T
    print('inverting H...')
    H_inv = torch.linalg.inv(H)
    del H
    torch.cuda.empty_cache()
    
    return H_inv



###### NTK inference helper ######
def kernel_inf(model,H_inv,x_train,test_set,per_class_n_train:int,block_size:int=0,n_classes:int=10,log_every:int=10,dry_run:int=0):
    
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    
    n_correct = 0.
    fnet, params = make_functional(model)
    if block_size > 0:
        assert (per_class_n_train*n_classes) % block_size == 0, "per_class_n_train must be divisible by block_size."
        assert per_class_n_train // block_size < 100, "number of blocks must be less than 100."

    with torch.no_grad():
        for idx, (x,y) in enumerate(test_set):
            if block_size == 0:
                a = empirical_ntk(fnet_single, params, x_train, x.unsqueeze(1).float().to(device), 'trace')
            else:
                a = torch.zeros(per_class_n_train*n_classes,1,device=device)
                for i in range(0,(per_class_n_train*n_classes)//block_size):
                    a[i*block_size:(i+1)*block_size,:] = empirical_ntk(fnet_single, params, x_train[i*block_size:(i+1)*block_size], x.unsqueeze(1).float().to(device), 'trace')
            result = a.T@H_inv
            n_correct += 1. if (result.argmax().item() // per_class_n_train) == y else 0.
            del result
            torch.cuda.empty_cache()
            if idx%log_every == log_every-1:
                print(f'({idx+1}): n_correct = {n_correct} || running acc. = {n_correct / (idx+1):.5f}')
            sys.stdout.flush()    
            if dry_run == 1 and idx == 5*log_every:
                break
    return n_correct




def main(args):
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create dir. to save all files.
    os.makedirs(args.save_base_dir, exist_ok=True)
    
    # Model init.
    if args.model_type == "MLP":
        model = MLP(784,args.hidden_size,args.hidden_size,args.hidden_size,10).to(device)
    elif args.model_type == "CNN":
        model = CNN(args.hidden_size,args.hidden_size,args.hidden_size).to(device)
    else:
        raise NotImplementedError(args.model_type)
    print(model)
    print(summary(model))
    print('\n')
        
    # Dataset init.
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root='.', train=True, download=True, transform=data_transform)
    test_set = datasets.MNIST(root='.', train=False, download=True, transform=data_transform)
    
    # Kernel databse init.
    per_class_n_train = args.ntk_train_samples
    per_class_n_test = args.ntk_test_samples

    train_idx, _ = divide_dataset(train_set, args.ntk_train_samples)
    test_idx,_  = divide_dataset(test_set, args.ntk_test_samples)

    train_data,train_label = train_set.data[train_idx], train_set.targets[train_idx]
    test_data,test_label = test_set.data[test_idx], test_set.targets[test_idx]

    x_train = train_data.unsqueeze(1).float().to(device)
    x_test = test_data.unsqueeze(1).float().to(device)
    
    save_dir = os.path.join(args.save_base_dir,f'{model.__class__.__name__}_h_{args.hidden_size}_n_{args.ntk_train_samples}_b_{args.ntk_train_block_size}')
    os.makedirs(save_dir, exist_ok=True)
    
    
    ntk_stats = {'lazy':[],'mf':[],}
    # Lazy regime:
    if args.regimes == "both" or args.regimes == "ntk":
        save_dir_ntk = os.path.join(save_dir,'ntk')
        os.makedirs(save_dir_ntk, exist_ok=True)
        H_inv = get_H_inv(model, x_train, args.ntk_train_samples, args.ntk_train_block_size, args.hidden_size, save_dir = save_dir_ntk)
        print(f'\nLazy regime NTK inference:')
        n_correct = kernel_inf(model,H_inv,x_train,test_set,per_class_n_train,args.ntk_train_block_size,log_every = args.log_every, dry_run = args.dry_run)
        del H_inv
        torch.cuda.empty_cache()
        print(f'Lazy regime n_correct = {n_correct}')
        print(f'Lazy regime test acc. = {n_correct/len(test_set) if args.dry_run == 0 else n_correct/(5*args.log_every)}')
        ntk_stats['lazy'].append(n_correct)
        
    # Feature learning regime:
    if args.regimes == "both" or args.regimes == "mf":
        print(f'\nFeature learing regime:')
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1,gamma=0.5)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader  = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        best_acc, stats = train(model,optimizer,scheduler,train_loader,test_loader,hidden_size = args.hidden_size,num_epochs = args.max_epochs,patience = args.early_stop,save_dir=save_dir)
        print(f'best test acc. = {best_acc}\n')
        pickle.dump(stats, open(os.path.join(save_dir, f"{model.__class__.__name__}_{args.hidden_size}_train_stats.pkl"), "wb"))
        
        model_dict = torch.load(os.path.join(save_dir, f'{model.__class__.__name__}_{args.hidden_size}.pt'))
        model.load_state_dict(model_dict["model_dict"])
        
        save_dir_mf = os.path.join(save_dir,'mf')
        os.makedirs(save_dir_mf, exist_ok=True)
        
        H_inv = get_H_inv(model, x_train, args.ntk_train_samples, args.ntk_train_block_size, args.hidden_size, save_dir = save_dir_mf)
        print(f'\nFeature learing regime NTK inference:')
        n_correct = kernel_inf(model,H_inv,x_train,test_set,per_class_n_train,args.ntk_train_block_size,log_every = args.log_every,dry_run = args.dry_run)
        
        del H_inv
        torch.cuda.empty_cache()
        print(f'Feature learing regime n_correct = {n_correct}')
        print(f'Feature learing regime test acc. = {n_correct/len(test_set) if args.dry_run == 0 else n_correct/(5*args.log_every)}')
        ntk_stats['mf'].append(n_correct)
    
    pickle.dump(ntk_stats, open(os.path.join(save_dir, f"{model.__class__.__name__}_{args.hidden_size}_NTK_stats.pkl"), "wb"))
    json.dump(args.__dict__, open(os.path.join(save_dir, f"{model.__class__.__name__}_{args.hidden_size}_args.txt"), "w"), indent=2)

if __name__ == "__main__":
    import argparse, os
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-base-dir", type=str, default="/scratch/ec2684/neural-tangents/runs")
    parser.add_argument("--model-load-dir", type=str, default=None)
    parser.add_argument("--regimes", type=str, default="both", choices=["ntk","mf","both"])
    parser.add_argument("--model-type", type=str, default="MLP", choices=["MLP", "CNN",],)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ntk-train-samples", type=int, default=30)
    parser.add_argument("--ntk-test-samples", type=int, default=1)
    parser.add_argument("--ntk-train-block-size", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", type=int, default=0, choices=[0, 1])
    # -- for enabling weights and biases
    # parser.add_argument("--wandb", type=int, default=1, choices=[0, 1], help="Set 1 to use wandb.")
    
    args = parser.parse_args()
    main(args)
