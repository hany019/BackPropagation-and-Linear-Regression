activations = ['sigmoid','tanh']
epochs = [500,1000,5000] #number of epochs
n = [0.02,0.05,0.1] # learning rate [0.01 - 0.2]
alpha = [0.5,0.9] # momentum [0.1 - 0.9]
L = [4,5,6] #number of layers
cpt = 0
f = open('params.csv', 'w', encoding='utf-8')
f.write('Id,Activation,Epochs,Learning,Momentum,Layers\n')
for act in activations:
    for ep in epochs:
        for r in n:
            for m in alpha:
                for l in L:
                    f.write('{},{},{},{},{},{}\n'.format(cpt,act,ep,r,m,l))
                    cpt += 1
f.close()
