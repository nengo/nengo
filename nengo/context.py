from collections import deque

class ContextStack(deque):
    def add_to_current(self, obj):
        try:
            curr = self.__getitem__(-1)
        except IndexError:
            raise IndexError("Context has not been set")
        
        if not hasattr(curr, "add"):
            raise AttributeError("Current context has no add function")
        
        curr.add(obj)
