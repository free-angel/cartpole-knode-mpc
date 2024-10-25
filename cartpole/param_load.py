import torch

pthfile='./hybrid_model_cartpole.pth'
net = torch.load(pthfile)

##对模型进行参数使用
print(net)

#print(type(net))  # 类型是 dict
print(len(net))   # 长度为 8，即存在8个 key-value 键值对


for k in net.keys():
    print("k:",k)
    # s = k.split('.')
    # #print(s[0])
    # s = '{}_{}'.format(s[0],s[1][0])
    # print("s:", s)
    s1 = net[k].cpu().numpy()
    print("s1:",s1.shape,s1)
