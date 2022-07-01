#!/usr/bin/env python3.7

# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csr_matrix

max_time = 5*60

# def mosek_linprog_sparse_input(c, A_sparse, blc, buc, blx, bux, bkc, bkx, flag_max = False, flag_int = False, flag_initial_guess = False, initial_guess = np.array([])):

def solve_binary_programming_gurobi(c, A_sparse, blc, buc, bkc,  flag_max = False):
    #bkc mapping  k2v = {"fr": 3,"fx": 2,"lo":0,"ra":4,"up":1}
    #A_sparse is a row-dominant sparse matrix
    #c is an numpy array, whose shape is equal to the number of variables
    flag_opt_fea = True
    x = []
    try:

        # Create a new model
        # m = gp.Model("mip1")
        m = gp.Model("binaryprogramming")
        m.setParam('TimeLimit', max_time)

        #set as dual complex:
        m.setParam('Method', 3)
        m.setParam('Threads', 2)

        # Create variables
        # x = m.addVar(vtype=GRB.BINARY, name="x")
        # x = m.addVar(vtype=GRB.BINARY)
        # y = m.addVar(vtype=GRB.BINARY)
        # z = m.addVar(vtype=GRB.BINARY)

        vars = []
        for i in range(c.shape[0]):
            vars.append(m.addVar(vtype=GRB.BINARY))

        # Set objective
        # m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
        obj = None
        flag_none = True
        for i in range(c.shape[0]):
            if flag_none:
                flag_none = False
                obj = vars[i] * c[i]
            else:
                obj = obj + vars[i] * c[i]
        
        if flag_max:
            m.setObjective(obj, GRB.MAXIMIZE)
        else:
            m.setObjective(obj, GRB.MINIMIZE)
        
    
        indices = A_sparse.indices
        indptr = A_sparse.indptr
        data = A_sparse.data
        # print("indptr shape: ", indptr.shape[0])
        # print ("bkc: ", len(bkc))
        assert(indptr.shape[0] - 1 == len(bkc))
        for i in range(indptr.shape[0] - 1):
            cons = None
            flag_none = True
            for j in range(indptr[i], indptr[i + 1]):
                # onesub.append(indices[j])
                # oneval.append(data[j])
                if flag_none:
                    flag_none = False
                    cons = data[j] * vars[indices[j]]
                else:
                    cons = cons + data[j] * vars[indices[j]]
            if bkc[i] == 4 or bkc[i] == 1:
                m.addConstr(cons <= buc[i])
            if bkc[i] == 4 or bkc[i] == 0:
                m.addConstr(cons >= buc[i])
            if bkc[i] == 2:
                m.addConstr(cons == buc[i])
                


        # # Add constraint: x + 2 y + 3 z <= 4
        # m.addConstr(x + 2 * y + 3 * z <= 4)

        # # Add constraint: x + y >= 1
        # m.addConstr(x + y >= 1)

        # Optimize model
        m.optimize()

        for v in m.getVars():
            # print('%s %g' % (v.varName, v.x))
            x.append(v.x)

        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        flag_opt_fea = False
    except AttributeError:
        print('Encountered an attribute error')
        flag_opt_fea = False
    return (flag_opt_fea, np.array(x))


def solve_linear_programming_gurobi(c, A_sparse,blx, bux, blc, buc, bkc, flag_max = False):
    #bkc mapping  k2v = {"fr": 3,"fx": 2,"lo":0,"ra":4,"up":1}
    #A_sparse is a row-dominant sparse matrix
    #c is an numpy array, whose shape is equal to the number of variables
    flag_opt_fea = True
    x = []
    try:

        # Create a new model
        # m = gp.Model("mip1")
        m = gp.Model("linearprogramming")


        # Create variables
        # x = m.addVar(vtype=GRB.BINARY, name="x")
        # x = m.addVar(vtype=GRB.BINARY)
        # y = m.addVar(vtype=GRB.BINARY)
        # z = m.addVar(vtype=GRB.BINARY)

        vars = []
        for i in range(c.shape[0]):
            # vars.append(m.addVar(vtype=GRB.BINARY))
            vars.append(m.addVar(lb = blx[i], ub = bux[i]))

        # Set objective
        # m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
        obj = None
        flag_none = True
        for i in range(c.shape[0]):
            if flag_none:
                flag_none = False
                obj = vars[i] * c[i]
            else:
                obj = obj + vars[i] * c[i]
        
        if flag_max:
            m.setObjective(obj, GRB.MAXIMIZE)
        else:
            m.setObjective(obj, GRB.MINIMIZE)
        
    
        indices = A_sparse.indices
        indptr = A_sparse.indptr
        data = A_sparse.data
        # print("indptr shape: ", indptr.shape[0])
        # print ("bkc: ", len(bkc))
        assert(indptr.shape[0] - 1 == len(bkc))
        for i in range(indptr.shape[0] - 1):
            cons = None
            flag_none = True
            for j in range(indptr[i], indptr[i + 1]):
                # onesub.append(indices[j])
                # oneval.append(data[j])
                if flag_none:
                    flag_none = False
                    cons = data[j] * vars[indices[j]]
                else:
                    cons = cons + data[j] * vars[indices[j]]
            if bkc[i] == 4 or bkc[i] == 1:
                m.addConstr(cons <= buc[i])
            if bkc[i] == 4 or bkc[i] == 0:
                m.addConstr(cons >= buc[i])
            if bkc[i] == 2:
                m.addConstr(cons == buc[i])
                


        # # Add constraint: x + 2 y + 3 z <= 4
        # m.addConstr(x + 2 * y + 3 * z <= 4)

        # # Add constraint: x + y >= 1
        # m.addConstr(x + y >= 1)

        # Optimize model
        m.optimize()

        for v in m.getVars():
            # print('%s %g' % (v.varName, v.x))
            x.append(v.x)

        print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        flag_opt_fea = False
    except AttributeError:
        print('Encountered an attribute error')
        flag_opt_fea = False
    return (flag_opt_fea, np.array(x))



if __name__ == "__main__":
    c = np.array([1,1,2])
    rows = []
    cols = []
    datas = []
    cur_row = 0
    blc = []
    bkc = []
    buc = []

    rows.append(cur_row)
    cols.append(0)
    datas.append(1)
    rows.append(cur_row)
    cols.append(1)
    datas.append(2)
    rows.append(cur_row)
    cols.append(2)
    datas.append(3)
    buc.append(4)
    blc.append(0)
    bkc.append(1)
    cur_row += 1

    rows.append(cur_row)
    cols.append(0)
    datas.append(1)
    rows.append(cur_row)
    cols.append(1)
    datas.append(1)
    buc.append(0)
    blc.append(1)
    bkc.append(0)
    cur_row += 1

    A_sparse = csr_matrix((datas, (rows, cols)), shape=(cur_row, c.shape[0]))
    

    (flag_opt_fea, x) = solve_binary_programming_gurobi(c, A_sparse, blc, buc, bkc,  flag_max = True)
    