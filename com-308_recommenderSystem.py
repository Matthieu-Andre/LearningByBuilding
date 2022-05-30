from math import sqrt

MISSING = 'x'



def valid(rating): return rating >= 0

def nb_pairs(matrix):
    pairs = 0
    for line in matrix:
        for element in line:
            if( valid(element)):
                pairs += 1
    return pairs

class Utils:
    @staticmethod
    def parse(line):
        parsed = []
        for element in line:
            if( element == '−' or element == 'x'):
                rating = MISSING
            else:
                rating = int(element)
            parsed.append(rating)
        return parsed

class Vector:

    def __init__(self, values):
        self.v = Utils.parse(values)
        self.size = len(self.v)

    def add(self,u): 
        if( not isinstance(u, Vector)):
            u = Vector([u]*self.size)

        new_v = []
        for x,y in zip(u.v, self.v):
            new_v.append(x + y)
        return Vector(new_v)

    def sum(self):
        total = 0
        for value in self.v:
            total += value
        return total

    def minus(self):
        return [-x for x in self.v]
    
    def get(self, x):
        return self.v[x]


class Matrix:
    

    def __init__(self, values):
        if(isinstance(values, Matrix)):
            self.vectors = values.vectors
            self.shape = values.shape
        else:
            self.vectors = []
            for i, vector in enumerate(values):
                self.vectors.append ( Vector(vector))
            
            self.shape = {'m':len(self.vectors), 'n':self.vectors[0].size} 

    def average(self):
        total = 0
        for vector in self.vectors:
            total += vector.sum()
        return 1.0 * total / nb_pairs(self.vectors)

    def get(self, x,y):
        return self.vectors[x].get(y)

    def get_column(self, index):
        return [ x.get(index) for x in self.vectors]

    def transpose(self):
        transposed = []
        for i in range(self.shape.get('n')):
            transposed.append( self.get_column(i))
        return Matrix(transposed) 
      

    def show(self):
        for vector in self.vectors:
            print("| ", end='')
            for i in range(vector.size):
                current = vector.get(i)
                if( isinstance(current,str)):
                    print("  --   ", end='')
                else:
                    print("%6.2f " % current, end='')
            print('|')
        print()

    def __str__(self):
        pr = ''
        for vector in self.vectors:
            pr += "| "
            for i in range(vector.size):
                current = vector.get(i)
                if( isinstance(current,str)):
                    pr+="  --   "
                else:
                    pr += "%6.2f " % current
            pr +='|\n'
        pr+= '\n'
        return pr


    # def RMSE(self, other):


    #     rmse_vector = []
    #     for i in range(len(real)):
    #         for j in range(len(real[0])):
    #             if( valid(real[i][j])):
    #                 rmse_vector.append( ( real[i][j] - predicted[i][j])**2 )
    #     return sqrt(sum(rmse_vector)/ nb_pairs(real))


def swap( matrix, i, j):
    tmp = matrix.vectors[i]
    matrix.vectors[i] = matrix.vectors[j]
    matrix.vectors[j] = tmp
    return matrix

def add(m, i, j):
    """
    Add row i to row j
    """
    new_j = m.vectors[i].add(m.vectors[j])
    m.vectors[j] = new_j
    return m 

def leftmost(m):
    

def triangulize(m):










def RMSE(real, predicted):
    rmse_vector = []
    for i in range(len(real)):
        for j in range(len(real[0])):
            if( valid(real[i][j])):
                rmse_vector.append( ( real[i][j] - predicted[i][j])**2 )
    return sqrt(sum(rmse_vector)/ nb_pairs(real))





# -------------------------------------------------------------------------

with ( open('R.txt', 'r')) as f:
    data  = f.readlines()

# matrix = [ Utils.parse(line.split()) for line in data]
formatted_data = [ line.split() for line in data]
matrix = Matrix( formatted_data )
# print(matrix)   
# print(matrix.transpose())
# matrix.show()
# new = matrix.transpose()
# print(id(matrix))
# print(id(new))
# new.show()
# new.show()
# r_mean = average(matrix)

m = Matrix( [[1, 2, 1, 2], [3,3,3,3], [4,5,6,7] ])
print(m)
print(add(m, 1,2))


