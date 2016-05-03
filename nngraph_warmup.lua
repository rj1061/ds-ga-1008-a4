require 'nngraph'
require 'nn'


idx=nn.Identity()()
idy=nn.Identity()()
idz=nn.Identity()()

tanx=nn.Tanh()(nn.Linear(4,2)(idx))
sigy=nn.Sigmoid()(nn.Linear(5,2)(idy))

sqx=nn.Square()(tanx)
sqy=nn.Square()(sigy)

xy=nn.CMulTable()({sqx,sqy})
a=nn.CAddTable()({xy,idz})

net=nn.gModule({idx,idy,idz},{a})

criterion=nn.MSECriterion()

x=torch.rand(4)
y=torch.rand(5)
z=torch.rand(2)

da=torch.Tensor(2):fill(1)


out=net:forward({x,y,z})
criterion:forward(out,da)

df_do=criterion:backward(out,da)
net:zeroGradParameters()
net:backward({x,y,z},df_do)
net:updateParameters(0.01)

out=net:forward({x,y,z})
