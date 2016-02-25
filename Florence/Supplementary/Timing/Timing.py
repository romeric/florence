from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print 'Finished %r sub-routine. Time elapsed till now is %2.4f seconds' % (f.__name__, te-ts)
        return result
    return wrap
