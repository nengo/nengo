import numpy as np

def gen_transform(dim_pre, dim_post, array_size=1, weight=1,
                          index_pre=None, index_post=None, transform=None):
        """Helper function used by :func:`Network.connect()` to create
        the `dim_pre` by `dim_post` transform matrix.

        Values are either 0 or *weight*. *index_pre* and *index_post*
        are used to determine which values are non-zero, and indicate
        which dimensions of the pre-synaptic ensemble should be routed
        to which dimensions of the post-synaptic ensemble.

        :param int dim_pre: first dimension of transform matrix
        :param int dim_post: second dimension of transform matrix
        :param int array_size: size of the network array
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
            transform = [[0] * dim_pre for i in range(dim_post * array_size)]

            # default index_pre/post lists set up *weight* value
            # on diagonal of transform
            
            # if dim_post * array_size != dim_pre,
            # then values wrap around when edge hit
            if index_pre is None:
                index_pre = range(dim_pre) 
            elif isinstance(index_pre, int):
                index_pre = [index_pre] 
            if index_post is None:
                index_post = range(dim_post * array_size) 
            elif isinstance(index_post, int):
                index_post = [index_post]

            for i in range(max(len(index_pre), len(index_post))):
                pre = index_pre[i % len(index_pre)]
                post = index_post[i % len(index_post)]
                transform[post][pre] = weight

        transform = np.array(transform)

        # reformulate to account for post.array_size
        if transform.shape == (dim_post * array_size, dim_pre):

            array_transform = [[[0] * dim_pre for i in range(dim_post)]
                               for j in range(array_size)]

            for i in range(array_size):
                for j in range(dim_post):
                    array_transform[i][j] = transform[i * dim_post + j]

            transform = array_transform

        return transform

def pstc(tau):
    return {type:"ExponentialPSC", pstc:tau}

def uniform(low, high):
    return {'type':'uniform', 'low':low, 'high':high}
