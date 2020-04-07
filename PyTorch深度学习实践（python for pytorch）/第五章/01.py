知识点：优化器Optim
1.  SGD（stochastic gradient descent）
    用法：optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum=0.9)
2.  RMSprop
    用法：keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=1e-06)
3.  Adagrad
    用法：keras.optimizers.Adagrad(lr=0.01,epsilon=1e-06)
4.  Adadelta
    用法：keras.optimizers.Adadelta(lr=1.0,rho=0.95,epsilon=1e-06)
5.  Adam
    用法：keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
6.  Adamax
    用法：keras.optimizers.Adamax(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
