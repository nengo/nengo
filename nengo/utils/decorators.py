"""Functions and classes for making better decorators.

Decorators implemented as function closures have important flaws:
    - __name__ and __doc__ are wrong
    - function argspec is wrong
    - function source code cannot be retrieved
    - cannot be applied on top of decorators implemented as descriptors

The classes in this file solve these problems. They are used by the
``decorator`` function, which is itself a decorator, which should be
applied to all decorator functions in Nengo. This makes decorators
easy to write, while avoiding the issues above.

The code in this file is a modified subset of the wrapt package,
available at https://pypi.python.org/pypi/wrapt.
For more details on the implementations herein, see the code comments
in the wrapt source at https://github.com/GrahamDumpleton/wrapt.
The code is included here under the terms of wrapt's BSD license.

----

Copyright (c) 2013, Graham Dumpleton. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import functools
import inspect

from nengo.utils.compat import with_metaclass


class ObjectProxyMethods(object):
    """Properly override __module__ and __doc__.

    If we add these in ObjectProxy, the derived class __dict__ will
    still be setup to have string variants of these attributes and the
    rules of descriptors means that they appear to take precedence
    over the properties in the base class. To avoid that, we copy the
    properties into the derived class type itself via a meta class. In
    that way the properties will always take precedence.
    """

    @property
    def __module__(self):
        return self.__wrapped__.__module__

    @property
    def __doc__(self):
        return self.__wrapped__.__doc__

    @property
    def __dict__(self):
        """Must be explicit to ensure that vars() works as expected."""
        return self.__wrapped__.__dict__


class ObjectProxyMeta(type):
    """Ensure ObjectProxyMethods take precedence.

    Copy our special properties into the class so that they always
    take precedence over attributes of the same name added during
    construction of a derived class. This is to save duplicating the
    implementation for them in all derived classes.
    """

    def __new__(cls, name, bases, dictionary):
        dictionary.update(vars(ObjectProxyMethods))
        return type.__new__(cls, name, bases, dictionary)


class ObjectProxy(with_metaclass(ObjectProxyMeta)):
    """A transparent object proxy for creating decorator descriptors.

    This is used in lieu of ``functools.update_wrapper``, which copies
    a number of properties of the wrapped function in the wrapper.
    Copying properties can be expensive though, so this is used instead
    to make the wrapper act like the wrapped function in all cases
    except ``__call__``.
    """

    __slots__ = '__wrapped__'

    def __init__(self, wrapped):
        object.__setattr__(self, '__wrapped__', wrapped)

        # Python 3 has the __qualname__ attribute, but it does not
        # allow it to be overridden using a property and it must instead
        # be an actual string object instead.
        try:
            object.__setattr__(self, '__qualname__', wrapped.__qualname__)
        except AttributeError:
            pass

    @property
    def __annotations__(self):
        return self.__wrapped__.__anotations__

    @property
    def __name__(self):
        return self.__wrapped__.__name__

    @property
    def __class__(self):
        return self.__wrapped__.__class__

    def __dir__(self):
        return dir(self.__wrapped__)

    def __getattr__(self, key):
        return getattr(self.__wrapped__, key)

    def __hash__(self):
        return hash(self.__wrapped__)

    def __setattr__(self, key, value):
        setattr(self.__wrapped__, key, value)

    def __str__(self):
        return str(self.__wrapped__)

    def __repr__(self):
        return '<%s at 0x%x for %s at 0x%x>' % (
            type(self).__name__, id(self),
            type(self.__wrapped__).__name__,
            id(self.__wrapped__))


class BoundFunctionWrapper(ObjectProxy):
    """A descriptor to emulate a bound function.

    This is used to create bound function decorators.
    It maintains all of the nice introspection that can normally
    be done on bound functions.
    """

    __slots__ = ('instance', 'wrapper', 'binding', 'parent')

    def __init__(self, wrapped, instance, wrapper, binding, parent):
        super(BoundFunctionWrapper, self).__init__(wrapped)
        # Using object.__setattr__ to subvert ObjectProxy.__setattr__
        object.__setattr__(self, 'instance', instance)
        object.__setattr__(self, 'wrapper', wrapper)
        object.__setattr__(self, 'binding', binding)
        object.__setattr__(self, 'parent', parent)

    def __get__(self, instance, owner):
        if self.instance is None and self.binding == 'function':
            descriptor = self.parent.__wrapped__.__get__(instance, owner)
            return BoundFunctionWrapper(
                descriptor, instance, self.wrapper, self.binding, self.parent)
        return self

    def __call__(self, *args, **kwargs):
        if self.binding == 'function':
            if self.instance is None:
                instance, args = args[0], args[1:]
                wrapped = functools.partial(self.__wrapped__, instance)
                return self.wrapper(wrapped, instance, args, kwargs)
            else:
                return self.wrapper(
                    self.__wrapped__, self.instance, args, kwargs)
        else:
            instance = getattr(self.__wrapped__, '__self__', None)
            return self.wrapper(self.__wrapped__, instance, args, kwargs)


class FunctionWrapper(ObjectProxy):
    """A descriptor to emulate a function.

    This is used to create function decorators.
    It maintains all of the nice introspection that can normally
    be done on functions.
    """

    __slots__ = ('wrapper', 'binding')

    def __init__(self, wrapped, wrapper):
        super(FunctionWrapper, self).__init__(wrapped)
        # Using object.__setattr__ to subvert ObjectProxy.__setattr__
        object.__setattr__(self, 'wrapper', wrapper)
        if isinstance(wrapped, classmethod):
            object.__setattr__(self, 'binding', 'classmethod')
        elif isinstance(wrapped, staticmethod):
            object.__setattr__(self, 'binding', 'staticmethod')
        else:
            object.__setattr__(self, 'binding', 'function')

    def __get__(self, instance, owner):
        wrapped = self.__wrapped__.__get__(instance, owner)
        return BoundFunctionWrapper(
            wrapped, instance, self.wrapper, self.binding, self)

    def __call__(self, *args, **kwargs):
        return self.wrapper(self.__wrapped__, None, args, kwargs)


def decorator(wrapper):
    """Decorates decorators.

    This imposes a particular style of writing descriptors.
    The descriptor must accept four positional arguments:

    - ``wrapped``: the function being wrapped
    - ``instance``: the instance that is bound to the function in the case of
      bound functions (None in the case of plain functions)
    - ``args``: the positional arguments passed to the wrapped function
    - ``kwargs``: the keyword arguments passed to the wrapped function

    Examples
    --------

    Decorating a normal function (i.e., instance will always be None)::

        @decorator
        def my_decorator(wrapped, instance, args, kwargs):
            return wrapped(*args, **kwargs)

        @my_decorator
        def f():
            return 1

    Decorating a bound function::

        @decorator
        def my_decorator(wrapped, instance, args, kwargs):
            return wrapped(*args, **kwargs)

        class MyClass(object):
            @my_decorator
            def f(self):
                return self.num + 1
    """
    def _wrapper(wrapped, instance, args, kwargs):
        def _execute(wrapped, *_args, **_kwargs):
            if instance is None:
                target_wrapper = wrapper
            elif inspect.isclass(instance):
                target_wrapper = wrapper.__get__(None, instance)
            else:
                target_wrapper = wrapper.__get__(instance, type(instance))
            return FunctionWrapper(wrapped, target_wrapper)
        return _execute(*args, **kwargs)
    return FunctionWrapper(wrapper, _wrapper)
