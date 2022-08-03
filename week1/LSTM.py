import numpy as np
import pandas as pd
def sigmoid(x):
    return 1 / (1 +np.exp(-x))
def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

class LSTM:

    def Cell_forward(self, xt, a_prev, c_prev, parameters):
        Wf = parameters["Wf"] # forget gate weight
        bf = parameters["bf"]
        Wi = parameters["Wi"] # update gate weight (notice the variable name)
        bi = parameters["bi"] # (notice the variable name)
        Wc = parameters["Wc"] # candidate value weight
        bc = parameters["bc"]
        Wo = parameters["Wo"] # output gate weight
        bo = parameters["bo"]
        Wy = parameters["Wy"] # prediction weight
        by = parameters["by"]
        
        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape
    
        ### START CODE HERE ###
        # Concatenate a_prev and xt (≈1 line)
        concat = np.concatenate((a_prev, xt), axis=0)
    
        # Compute values for ft (forget gate), it (update gate),
        # cct (candidate value), c_next (cell state), 
        # ot (output gate), a_next (hidden state) (≈6 lines)
        ft = sigmoid(Wf.dot(concat) + bf)        # forget gate
        it = sigmoid(Wi.dot(concat) + bi)        # update gate
        cct = np.tanh(Wc.dot(concat) + bc)       # candidate value
        c_next = ft*c_prev + it*cct    # cell state
        ot = sigmoid(Wo.dot(concat) + bo)        # output gate
        a_next = ot*np.tanh(c_next)    # hidden state
        
        # Compute prediction of the LSTM cell (≈1 line)
        yt_pred = softmax(Wy.dot(a_next) + by)
        ### END CODE HERE ###
    
        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    
        return a_next, c_next, yt_pred, cache

    def forward(self, x, a0, parameters):
        # Initialize "caches", which will track the list of all the caches
        caches = []
        
        ### START CODE HERE ###
        Wy = parameters['Wy'] # saving parameters['Wy'] in a local variable in case students use Wy instead of parameters['Wy']
        # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
        n_x, m, T_x = x.shape
        n_y, n_a = Wy.shape
        
        # initialize "a", "c" and "y" with zeros (≈3 lines)
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))
        
        # Initialize a_next and c_next (≈2 lines)
        a_next = a0
        c_next = np.zeros((n_a, m))
        
        # loop over all time-steps
        for t in range(T_x):
            # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
            xt = x[:,:,t]
            # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
            a_next, c_next, yt, cache = self.Cell_forward(xt, a_next, c_next, parameters)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:,:,t] = a_next
            # Save the value of the next cell state (≈1 line)
            c[:,:,t]  = c_next
            # Save the value of the prediction in y (≈1 line)
            y[:,:,t] = yt
            # Append the cache into caches (≈1 line)
            caches.append(cache)
            
        ### END CODE HERE ###
        
        # store values needed for backward propagation in cache
        caches = (caches, x)
    
        return a, y, c, caches

    def Cell_backward(self, da_next, dc_next, cache):

 
        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
        
        ### START CODE HERE ###
        # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
        n_x, m = xt.shape
        n_a, m = a_next.shape
        
        # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
        dot = da_next * np.tanh(c_next) * ot * (1-ot)
        dcct = (dc_next * it + ot * (1-np.square(np.tanh(c_next))) * it * da_next) * (1-np.square(cct))
        dit = (dc_next * cct + ot*(1-np.square(np.tanh(c_next)))*cct*da_next) * it * (1-it)
        dft = (dc_next * c_prev + ot*(1-np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1-ft)
        
        # Compute parameters related derivatives. Use equations (11)-(18) (≈8 lines)
        dWf = np.dot(dft,np.concatenate((a_prev, xt), axis=0).T)
        dWi = np.dot(dit,np.concatenate((a_prev, xt), axis=0).T)
        dWc = np.dot(dcct,np.concatenate((a_prev, xt), axis=0).T)
        dWo = np.dot(dot,np.concatenate((a_prev, xt), axis=0).T)
        dbf = np.sum(dft,axis=1,keepdims=True)
        dbi = np.sum(dit,axis=1,keepdims=True)
        dbc = np.sum(dcct,axis=1,keepdims=True)
        dbo = np.sum(dot,axis=1,keepdims=True)
    
        # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (19)-(21). (≈3 lines)
        da_prev = np.dot(parameters['Wf'][:,:n_a].T,dft)+np.dot(parameters['Wi'][:,:n_a].T,dit)+np.dot(parameters['Wc'][:,:n_a].T,dcct)+np.dot(parameters['Wo'][:,:n_a].T,dot)
        dc_prev = dc_next*ft+ot*(1-np.square(np.tanh(c_next)))*ft*da_next
        dxt = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot)
        ### END CODE HERE ###
        
        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
        return gradients

    def backward(self, da, caches):
    
        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
        
        ### START CODE HERE ###
        # Retrieve dimensions from da's and x1's shapes (≈2 lines)
        n_a, m, T_x = da.shape
        n_x, m = x1.shape
        
        # initialize the gradients with the right sizes (≈12 lines)
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))
        dc_prevt = np.zeros((n_a, m))
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros((n_a, n_a + n_x))
        dWc = np.zeros((n_a, n_a + n_x))
        dWo = np.zeros((n_a, n_a + n_x))
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros((n_a, 1))
        dbc = np.zeros((n_a, 1))
        dbo = np.zeros((n_a, 1))
        
        # loop back over the whole sequence
        for t in reversed(range(T_x)):
            # Compute all gradients using lstm_cell_backward
            gradients = self.Cell_backward(da[:,:,t]+da_prevt, dc_prevt, caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            da_prevt = gradients["da_prev"]
            dc_prevt = gradients["dc_prev"]
            dx[:,:,t] = gradients["dxt"]
            dWf = dWf+gradients["dWf"]
            dWi = dWi+gradients["dWi"]
            dWc = dWc+gradients["dWc"]
            dWo = dWo+gradients["dWo"]
            dbf = dbf+gradients["dbf"]
            dbi = dbi+gradients["dbi"]
            dbc = dbc+gradients["dbc"]
            dbo = dbo+gradients["dbo"]
        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients["da_prev"]
        
        ### END CODE HERE ###
    
        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
        
        return gradients