import logging
logger = logging.getLogger(__name__)

_stack = []

def add_to_current(obj):
    """Add an object to the current context."""
    
    con = current()
    if con is None:
        raise AttributeError("Can't add object without setting context")
    else:
        con.add(obj)

def current():
    """Return the current context."""
    
    global _stack
    
    return _stack[-1] if len(_stack) > 0 else None

def push(con):
    """Add a new context to the top of the stack."""
    
    global _stack
    
    if not isinstance(con, Context):
        raise TypeError("Only Context objects can be added to context stack")
    elif current() != con:
        _stack += [con]
        
    if len(_stack) > 100:
        logger.warning("Context stack is getting quite large, this is probably not intended")

def pop():
    """Remove a context from the stack."""
    
    global _stack
    _stack.pop()
    
def clear():
    """Empties the stack."""
    
    global _stack
    _stack = []
    
class Context:
    """Any object that wants to be able to set itself as a context should subclass off this."""
    
    def add(self, obj):
        #override this in subclass
        raise NotImplementedError
    
    def __enter__(self):
        push(self)
        
    def __exit__(self, exception_type, exception_value, traceback):
        pop()
