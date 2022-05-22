import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import kl_divergence, Normal
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from torch import optim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

BATCH_SIZE = 128
NUM_WORKERS = 0
NUM_EPOCHS = 200
lr = 1e-3
input_dims = 12
hidden_dims = 1024
z1_dims = 128
num_steps = 32
is_cuda = True
decay = 0.9999
save_path = './ckpt-PVAE/'

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

class PurifiedVAE(nn.Module):
    def __init__(self, input_dims, hidden_dims, z1_dims, n_step, k=1000, num_layers=1, is_bidirection=True, is_training=True, is_cuda=False):
        # z1: chroma+root
        super(PurifiedVAE, self).__init__()
        self.input_dims = input_dims # only chroma
        self.hidden_dims = hidden_dims
        self.z1_dims = z1_dims
        self.n_step = n_step
        self.eps = 1
        self.sample = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])

        self.training = is_training
        self.is_cuda = is_cuda


        self.hidden_factor = (2 if is_bidirection else 1) * num_layers
        self.gru_0 = nn.GRU(input_dims, hidden_dims, batch_first=True, bidirectional=is_bidirection)
        self.grucell_0 = nn.GRUCell(input_dims + z1_dims, hidden_dims)
        self.mu = nn.Linear(hidden_dims * self.hidden_factor, z1_dims)
        self.var = nn.Linear(hidden_dims * self.hidden_factor, z1_dims)
        self.linear_init_0 = nn.Linear(z1_dims, hidden_dims)
        self.linear_out_0 = nn.Linear(hidden_dims, input_dims)
        self.sigmoid = nn.Sigmoid()


    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x):
        _, x = self.gru_0(x)
        x = x.transpose_(0,1).contiguous()
        x = x.view(x.size(0), -1)
        mean = self.mu(x)
        stddev = (self.var(x) * 0.5).exp_()
        return Normal(mean, stddev)

    def decoder(self, z):
        out = torch.zeros((z.size(0), self.input_dims)) # GRUcell's input and output
        #print(out.shape)
        out[:, -1] = 1 # ?
        x = [] # final output
        #print(z.size())
        t = torch.tanh(self.linear_init_0(z))
        hx = t
        if self.is_cuda and torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out, z], 1) # batch_size * (input_dims + z1_dims)
            #print(out.shape)
            hx = self.grucell_0(out, hx)
            out = self.sigmoid(self.linear_out_0(hx)) # batch_size * input_dims
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / (self.k + torch.exp(self.iteration / self.k))
                self.iteration += 1
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x):
        if self.training:
            self.sample = x.clone()
        latent = self.encoder(x)
        if self.training:
            z = latent.rsample()
        else:
            z = latent.mean
        return self.decoder(z), latent.mean, latent.stddev



def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N

def loss_function(recon, target_tensor, distribution, beta=.1):
    BCE = F.binary_cross_entropy(recon, target_tensor, reduction='elementwise_mean')
    normal = std_normal(distribution.mean.size())
    KL = kl_divergence(distribution, normal).mean()
    return BCE + beta * KL

model = PurifiedVAE(input_dims, hidden_dims, z1_dims, num_steps, is_cuda=is_cuda)
if model.is_cuda:
    model.cuda()
    #model.train()
    




bar8_cp_np = np.load('data/bar8_cp_np.npy')
bar8_cp_wq_np = np.load('data/bar8_cp_with_quality_np.npy')
bar8_comb = []

for i in range(6000):
    bar8_comb.append([bar8_cp_np[i], bar8_cp_wq_np[i]]) # data, label
    
for i in range(6000):
    bar8_comb.append([bar8_cp_wq_np[i], bar8_cp_wq_np[i]]) # data, label

for i in range(6000, bar8_cp_np.shape[0]):
    bar8_comb.append([bar8_cp_np[i], bar8_cp_wq_np[i]]) # data, label
    
for i in range(6000, bar8_cp_np.shape[0]):
    bar8_comb.append([bar8_cp_wq_np[i], bar8_cp_wq_np[i]]) # data, label

bar8_cp_tensor = torch.tensor(bar8_cp_np, dtype=torch.float32)
bar8_cp_wq_tensor = torch.tensor(bar8_cp_wq_np, dtype=torch.float32)
bar8_comb_tensor = torch.tensor(bar8_comb, dtype=torch.float32)

basic_trainset = bar8_cp_wq_tensor[:6000]
basic_testset = bar8_cp_wq_tensor[6000:]
full_trainset = bar8_comb_tensor[:6000]
full_testset = bar8_comb_tensor[12000:12451]
comb_trainset = bar8_comb_tensor[:12000]
comb_testset = bar8_comb_tensor[12000:]

basic_trainloader = torch.utils.data.DataLoader(basic_trainset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=NUM_WORKERS)
basic_testloader = torch.utils.data.DataLoader(basic_testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=NUM_WORKERS)
full_trainloader = torch.utils.data.DataLoader(full_trainset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=NUM_WORKERS)
full_testloader = torch.utils.data.DataLoader(full_testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=NUM_WORKERS)
comb_trainloader = torch.utils.data.DataLoader(comb_trainset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=NUM_WORKERS)
comb_testloader = torch.utils.data.DataLoader(comb_testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=NUM_WORKERS)

def get_final(recon):
    return recon.apply_(lambda x: 1 if x>=0.5 else 0)

def test_recon(sources, targets):
    with torch.no_grad():
        model.eval()
        recons, _, _ = model(sources.cuda())
        recons = get_final(recons.cpu())
    size = recons.size()
    amount = 1
    for x in size:
        amount *= x
    return (torch.norm(recons-targets, 1)/amount).item()


# Stage 1

optimizer = optim.Adam(model.parameters(), lr)
scheduler = MinExponentialLR(optimizer, gamma=decay, minimum=1e-5)

writer = SummaryWriter('log/{}'.format('PurifiedVAE'))

lr = 1e-3
for epoch in range(NUM_EPOCHS//4):
    for i, targets in enumerate(basic_trainloader):  
        # Move tensors to the configured device
        if model.cuda:
            targets = targets.cuda()
            
        optimizer.zero_grad()
        
        # Forward pass
        recons, means, stddevs = model(targets)
        distribution = Normal(means, stddevs)
        loss = loss_function(recons, targets, distribution)
        
        # Backward and optimize
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        
        if i % 50 == 0:
            print('batch loss: {:.5f}'.format(loss.item()))
        writer.add_scalar('batch_loss', loss.item(), i)
        
        
    
    if epoch % 10 == 0:
        torch.save(model.cpu().state_dict(), save_path+'stage-1-'+str(epoch)+'-epoch_PurifiedVAE.ckpt')
        if torch.cuda.is_available():
            model.cuda()
        print(epoch, '-epoch Model saved!')
        print('train: ', test_recon(bar8_cp_tensor[:6000], bar8_cp_tensor[:6000]))
        print('test: ', test_recon(bar8_cp_tensor[6000:], bar8_cp_tensor[6000:]))
        model.train()
        
torch.cuda.empty_cache()
        
# Stage 2

for n, p in model.named_parameters():
    if n in {'grucell_0.weight_ih', 'grucell_0.weight_hh', 'grucell_0.bias_ih', 'grucell_0.bias_hh', 'linear_out_0.weight', 'linear_out_0.bias'}:
        p.requires_grad = False
    else:
        #print(n)
        p.requires_grad = True


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)
scheduler = MinExponentialLR(optimizer, gamma=decay, minimum=1e-8)

writer = SummaryWriter('log/{}'.format('PurifiedVAE'))

lr = 1e-3
for epoch in range(NUM_EPOCHS//4, NUM_EPOCHS):
    for i, data in enumerate(comb_trainloader):  
        # Move tensors to the configured device
        sources = data[:,0,:,:]
        targets = data[:,1,:,:]
        if model.cuda:
            sources = sources.cuda()
            targets = targets.cuda()
            
        optimizer.zero_grad()
        
        # Forward pass
        recons, means, stddevs = model(sources)
        distribution = Normal(means, stddevs)
        loss = loss_function(recons, targets, distribution)
        
        # Backward and optimize
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        
        if i % 50 == 0:
            print('batch loss: {:.5f}'.format(loss.item()))
        writer.add_scalar('batch_loss', loss.item(), i)
        
        
    
    if epoch % 10 == 0:
        torch.save(model.cpu().state_dict(), save_path+'stage-2-'+str(epoch)+'-epoch_PurifiedVAE.ckpt')
        if torch.cuda.is_available():
            model.cuda()
        print(epoch, '-epoch Model saved!')
        #print('train: ', test_recon(bar8_comb_tensor[:12000,0,:,:], bar8_comb_tensor[:12000,0,:,:]))
        print('test: ', test_recon(bar8_comb_tensor[12000:,0,:,:], bar8_comb_tensor[12000:,0,:,:]))
        model.train()
        
    