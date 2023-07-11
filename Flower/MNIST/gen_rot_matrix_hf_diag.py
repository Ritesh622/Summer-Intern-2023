import torch
from scipy.linalg import hadamard

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
data_type = torch.float16 if DEVICE.type=='cuda' else torch.float32
dims = int(input("dimensions of rotation matrix to generate: "))

had_mat = torch.tensor(hadamard(dims), dtype=data_type)
had_mat = had_mat.to(DEVICE)

diag_mat = torch.diag(torch.where(torch.cat((torch.ones(dims//2,),
     torch.randint(low=0, high=2, size=(dims//2, ))))==0, -1, 1)).type(data_type)
diag_mat = diag_mat.to(DEVICE)

rot_mat = torch.matmul(had_mat, diag_mat)
rot_mat = rot_mat/(dims**0.5)

save_loc = input("file name: ")
torch.save(rot_mat, save_loc+'.pt')
