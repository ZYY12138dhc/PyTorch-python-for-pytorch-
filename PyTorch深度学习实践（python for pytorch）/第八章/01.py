#保存整个神经网络的结构和模型参数
torch.save(mymodel.'mymodel.pkl')
#只保存神经网络的模型参数
torch.save(mymodel.state_dict(),'mymodel_params.pkl')
#导入模型
mymodel = torch,load('mymodel.pkl')
