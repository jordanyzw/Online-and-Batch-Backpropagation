# -*- coding: cp936 -*-
# Back-Propagation Neural Networks

from numpy import *
import random
import string

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function
def sigmoid(x):
    return 1.0/(1+exp(-x))

# derivative of our sigmoid function
# 在求取输出层和隐藏侧的误差项的时候会用到
def dsigmoid(y):
    return y*(1-y)

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        # 输入层，隐藏层，输出层的数量，三层网络
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        #生成权重矩阵，每一个输入层节点和隐藏层节点都连接
        #每一个隐藏层节点和输出层节点链接
        #大小：self.ni*self.nh
        self.ui = makeMatrix(self.ni, self.nh)
        self.vi = makeMatrix(self.ni, self.nh)
        #大小：self.ni*self.nh
        self.uo = makeMatrix(self.nh, self.no)
        self.vo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        #生成权重，在-0.2-0.2之间
        for i in range(self.ni):
            for j in range(self.nh):
                self.ui[i][j] = rand(-1.0, 1.0)
                self.vi[i][j] = rand(-1.0, 1.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.uo[j][k] = rand(-1.0, 1.0)
                self.vo[j][k] = rand(-1.0, 1.0)

        # last change in weights for momentum 
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):#对于一个样例输出结果
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        # 输入的激活函数s
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        #隐藏层的激活函数,求和然后使用压缩函数
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                #sum就是《ml》书中的net
                sum = sum + self.ai[i] * self.ai[i] * self.ui[i][j] + self.ai[i] * self.vi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        #输出的激活函数
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.ah[j] * self.uo[j][k] + self.ah[j] * self.vo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    #online反向传播算法 targets是样本的正确的输出,每次都更新权值
    def OnlinebackPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        #计算输出层的误差项 
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            #计算k-o
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        #计算隐藏层的误差项
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * ( 2*self.ah[j]*self.uo[j][k] + self.vo[j][k] )
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights，
        # 更新输出层的权重参数
        # 这里可以看出，本例使用的是带有“增加冲量项”的BPANN
        # 其中，N为学习速率 M为充量项的参数 self.co为冲量项
        # N: learning rate
        # M: momentum factor
        for j in range(self.nh):
            for k in range(self.no):
                uochange = output_deltas[k]*self.ah[j]*self.ah[j]
                vochange = output_deltas[k]*self.ah[j]
                self.uo[j][k] = self.uo[j][k] + N*uochange + M*self.co[j][k]
                self.vo[j][k] = self.vo[j][k] + N*vochange + M*self.co[j][k]
                self.co[j][k] = uochange + vochange
                #print N*change, M*self.co[j][k]

        # update input weights
        #更新输入项的权重参数
        for i in range(self.ni):
            for j in range(self.nh):
                uichange = hidden_deltas[j]*self.ai[i]*self.ai[i]
                vichange = hidden_deltas[j]*self.ai[i]
                self.ui[i][j] = self.ui[i][j] + N*uichange + M*self.ci[i][j]
                self.vi[i][j] = self.vi[i][j] + N*vichange + M*self.ci[i][j]
                self.ci[i][j] = uichange + vichange

        # calculate error
        #计算E(w)
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def BatchbackPropagate(self, datamat, targetmat, N,M):
        #对于输入层的权值更新
        n = len(targetmat)
        sumuo = [0.0]*self.no
        sumvo = [0.0]*self.no
        sumui = [0.0]*self.nh
        sumvi = [0.0]*self.nh
        for i in range(n):
            output_deltas = [0.0] * self.no
            self.update(datamat[i])#把样例输入到该网络，更新ai,ah,ao
            for j in range(self.nh):
                for k in range(self.no):                    
                    output_deltas[k] = dsigmoid(self.ao[k]) * (targetmat[i][k] - self.ao[k])
                    sumuo[k] = sumuo[k] +  output_deltas[k]*self.ah[j]*self.ah[j]
                    sumvo[k] = sumvo[k] +  output_deltas[k]*self.ah[j]          

            hidden_deltas = [0.0] * self.nh
            for j in range(self.nh):          
                error = 0.0
                for k in range(self.no):                    
                    error = error + output_deltas[k] * ( 2*self.ah[j]*self.uo[j][k] + self.vo[j][k])
                hidden_deltas[j] = dsigmoid(self.ah[j]) * error
            for j in range(self.ni):
                for k in range(self.nh):      
                    sumui[k] = sumui[k] + hidden_deltas[k]*self.ai[j]*self.ai[j]
                    sumvi[k] = sumvi[k] + hidden_deltas[k]*self.ai[j]
        alpha = 1          
        for i in range(self.nh):#更新输出端的权值
            for j in range(self.no):
                self.uo[i][j] = alpha*self.uo[i][j] + N * sumuo[j] + M*self.co[i][j]
                self.vo[i][j] = alpha*self.vo[i][j] + N * sumvo[j] + M*self.co[i][j]
                self.co[i][j] = sumuo[j] + sumvo[j]
        for i in range(self.ni):#更新隐藏层的权值
            for j in range(self.nh):
                self.ui[i][j] = alpha*self.ui[i][j] + N * sumui[j]+ M*self.ci[i][j]
                self.vi[i][j] = alpha*self.vi[i][j] + N * sumvi[j]+ M*self.ci[i][j]
                self.ci[i][j] = sumui[j] + sumvi[j]
        error = 0.0
        for i in range(n):
            self.update(datamat[i])
            for k in range(self.no):
                error = error + 0.5*(targetmat[i][k]-self.ao[k])**2
        return error
    def OnlineLearningTrain(self,iterations=1000, N=0.03, M=0.05):
        # N: learning rate
        # M: momentum factor
        atributemat = []; labelmat = []
        fr = open('train.txt')
        for line in fr.readlines():
            linearr = line.strip().split()
            atributemat.append([float(linearr[0]),float(linearr[1])])
            labelmat.append([int(linearr[2])])
        n = shape(atributemat)[0]
        
        for i in range(iterations):
            error = 0.0
            for j in range(n):
                inputs = atributemat[j]
                targets = labelmat[j]
                self.update(inputs)
                error = error + self.OnlinebackPropagate(targets, N, M)
            if i % 100 == 0:
                print"After" ,i, "iterations,the total error of the training set is",error
    def BatchLearningTrain(self, iterations=60000,N=0.01,M = 0.001):
        # N: learning rate
        # M: momentum factor
        atributemat = []; labelmat = []
        fr = open('train.txt')
        for line in fr.readlines():
            linearr = line.strip().split()
            atributemat.append([float(linearr[0]),float(linearr[1])])
            labelmat.append([int(linearr[2])])
        print"this algorithm needs much more time to get a error rate little then 30 percent" 
        for i in range(iterations):
            tmpdatamat = atributemat
            tmplabelmat = labelmat
            error = self.BatchbackPropagate(tmpdatamat, labelmat, N,M)
            if i % 1000 == 0:
                print"After" ,i, "iterations,the total error of the training set is",error
            i = i+1
    #测试函数，用于测试训练效果
    def test(self):
        atributemat = []; labelmat = []
        fr = open('test.txt')
        for line in fr.readlines():
            linearr = line.strip().split()
            atributemat.append([float(linearr[0]),float(linearr[1])])
            labelmat.append([int(linearr[2])])
        n = shape(atributemat)[0]
        errcount = 0
        for i in range(n):
            temp = self.update(atributemat[i])[0]
            print"the",i,"test example is",labelmat[i][0],"and the BP is ouput is",temp
            if((labelmat[i][0] == 1 and temp < 0.5) or (labelmat[i][0] == 0 and temp > 0.5)):
                print('case %d is not classified correctly' % i)
                errcount = errcount + 1
        print('the total error rate of the test set is %f' % (errcount/(n*1.0)))
        return (errcount/(n*1.0))
            
    def printweights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.ui[i])
            print(self.vi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.uo[j])
            print(self.vo[i])


def demo():
    # create a network with two input, one hidden,in the hidden there are two nueron, and one output nodes
    print '------------- OnlineLearningBP-------------'
    n1 = NN(2, 10, 1)
    n1.OnlineLearningTrain()
    n1.test()
    print '--------------BatchLearningBP--------------'
    n2 = NN(2,10,1)
    n2.BatchLearningTrain()
    n2.test()
    


if __name__ == '__main__':
    demo()
