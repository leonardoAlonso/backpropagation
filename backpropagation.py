import math
import random
import matplotlib.pyplot as plt


class NN:
    def __init__(self, NI, NH, NO):
        # numero de neuronas en las capas
        self.ni = NI + 1  # +1 for bias
        self.nh = NH
        self.no = NO

        # initialize node-activations
        self.ai, self.ah, self.ao = [], [], []
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # inicializar los persos de forma aleatoria
        randomizeMatrix(self.wi, -0.2, 0.2)
        randomizeMatrix(self.wo, -2.0, 2.0)
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

        #vector outputs
        self.outputs = []

    def runNN(self, inputs):
        if len(inputs) != self.ni - 1:
            print('incorrect number of inputs')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += (self.ai[i] * self.wi[i][j])
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += (self.ah[j] * self.wo[j][k])
            self.ao[k] = sigmoid(sum)
        return self.ao

    def backPropagate(self, targets, N, M):

        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * dsigmoid(self.ao[k])

            # actualizar los pesos de salida
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change + M * self.co[j][k]
                self.co[j][k] = change


        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])

        # actualizar pesos de la capa de entrada
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                # print 'activation',self.ai[i],'synapse',i,j,'change',change
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        error = 0.0
        for k in range(len(targets)):
            error = 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])
        print('')

    def test(self, patterns):
        for p in patterns:
            inputs = p[0]
            print('Inputs:', p[0], '-->', self.runNN(inputs), '\tTarget', p[1])
            self.outputs.append(self.runNN(inputs))

    def train(self, patterns, max_iterations=3500, N=0.5, M=0.1):
        errors  = []
        m = []
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N, M)
            #if i % 50 == 0:
            if error < 0.0001:
                break
        #self.test(patterns)
            errors.append(error)
        plt.plot(errors)
        plt.show()
        print(error)

def sigmoid(x):
    return  math.tanh(x)

def dsigmoid(y):
    return 1 - y ** 2


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


def randomizeMatrix(matrix, a, b):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)

def main():
    pat = [
        [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]],
    ]
    inputs = [
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]
    ]
    myNN = NN(25, 5, 25)
    myNN.train(pat[0][0])
    outputs = myNN.runNN(inputs)



if __name__ == "__main__":
    main()