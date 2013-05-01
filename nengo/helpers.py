import numpy as np
import math
import inspect

def gen_transform(dim_pre, dim_post, weight=1,
                          index_pre=None, index_post=None, transform=None):
        """Helper function used by :func:`Network.connect()` to create
        the `dim_pre` by `dim_post` transform matrix.

        Values are either 0 or *weight*. *index_pre* and *index_post*
        are used to determine which values are non-zero, and indicate
        which dimensions of the pre-synaptic ensemble should be routed
        to which dimensions of the post-synaptic ensemble.

        :param int dim_pre: first dimension of transform matrix
        :param int dim_post: second dimension of transform matrix
        :param float weight: the non-zero value to put into the matrix
        :param index_pre: the indexes of the pre-synaptic dimensions to use
        :type index_pre: list of integers or a single integer
        :param index_post:
            the indexes of the post-synaptic dimensions to use
        :type index_post: list of integers or a single integer
        :returns:
            a two-dimensional transform matrix performing
            the requested routing

        """

        if transform is None:
            # create a matrix of zeros
            transform = [[0] * dim_pre for i in range(dim_post)]

            # default index_pre/post lists set up *weight* value
            # on diagonal of transform
            
            # if dim_post != dim_pre,
            # then values wrap around when edge hit
            if index_pre is None:
                index_pre = range(dim_pre) 
            elif isinstance(index_pre, int):
                index_pre = [index_pre] 
            if index_post is None:
                index_post = range(dim_post) 
            elif isinstance(index_post, int):
                index_post = [index_post]

            for i in range(max(len(index_pre), len(index_post))):
                pre = index_pre[i % len(index_pre)]
                post = index_post[i % len(index_post)]
                transform[post][pre] = weight

        transform = np.array(transform)

        return transform

def pstc(tau):
    return {'type':"ExponentialPSC", pstc:tau}

def uniform(low, high):
    return {'type':'uniform', 'low':low, 'high':high}

def sample_pdf(params, size):
    if params["type"].lower() == "uniform":
        return np.random.uniform(size=size, low=params["low"], high=params["high"])
    elif params["type"].lower() == "gaussian":
        return np.random.normal(size=size, loc=params["mean"], scale=params["variance"])
    else:
        print "Unrecognized pdf type"
        
def fix_function(func):
    """Takes a given function and wraps it so that it is always returns an nparray"""
        
    #if it's a lambda function, change the name to "output" (for ease of use
    #when referring to the output later)
    if func.__name__ == "<lambda>":
        func.__name__ = "output"
        
    name = func.__name__
    
    python_func = func
    if isinstance(func, (type(math.sin),type(np.sin))):
        #this means func is a python or numpy builtin, so wrap it in a regular
        #python function so it can be inspected in other places with
        #the inspect module
        try:
            func()
            func = lambda : func()
        except (ValueError,TypeError):
            #this means the func() call failed (func doesn't accept
            #0 arguments)
            try:
                func(0.0)
                func = lambda t: func(t)
            except (ValueError,TypeError):
                print "Function must accept either 0 or 1 arguments"
                return None
        
    num_args = len(inspect.getargspec(python_func).args)
    
    array_func = python_func
            
    #check if it's returning a float or list rather than an nparray
    if num_args == 0:
        result = array_func()
        if isinstance(result, float):
            array_func = lambda : np.asarray([python_func()])
        elif isinstance(result, list):
            array_func = lambda : np.asarray(python_func())
    else:
        result = array_func(0.0)
        if isinstance(result, float):
            array_func = lambda t: np.asarray([python_func(t)])
        elif isinstance(result, list):
            array_func = lambda t: np.asarray(python_func(t))
    
    array_func.__name__ = name
    
    return array_func
