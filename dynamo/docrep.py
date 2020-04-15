# code copy from https://github.com/Chilipp/docrep/blob/master/docrep/__init__.py
# need to better properly include it

import types
import six
import inspect
import re
from warnings import warn


def dedents(s):
    warn(
        "The dedent function has been depreceated and will be removed soon. "
        "Use inspect.cleandoc instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return inspect.cleandoc(s)


__version__ = "0.2.7"

__author__ = "Philipp Sommer"


substitution_pattern = re.compile(
    r"""(?s)(?<!%)(%%)*%(?!%)   # uneven number of %
        \((?P<key>.*?)\)# key enclosed in brackets""",
    re.VERBOSE,
)


summary_patt = re.compile(r"(?s).*?(?=(\n\s*\n)|$)")


class _StrWithIndentation(object):
    """A convenience class that indents the given string if requested through
    the __str__ method"""

    def __init__(self, s, indent=0, *args, **kwargs):
        self._indent = "\n" + " " * indent
        self._s = s

    def __str__(self):
        return self._indent.join(self._s.splitlines())

    def __repr__(self):
        return repr(self._indent.join(self._s.splitlines()))


def safe_modulo(s, meta, checked="", print_warning=True, stacklevel=2):
    """Safe version of the modulo operation (%) of strings
    Parameters
    ----------
    s: str
        string to apply the modulo operation with
    meta: dict or tuple
        meta informations to insert (usually via ``s % meta``)
    checked: {'KEY', 'VALUE'}, optional
        Security parameter for the recursive structure of this function. It can
        be set to 'VALUE' if an error shall be raised when facing a TypeError
        or ValueError or to 'KEY' if an error shall be raised when facing a
        KeyError. This parameter is mainly for internal processes.
    print_warning: bool
        If True and a key is not existent in `s`, a warning is raised
    stacklevel: int
        The stacklevel for the :func:`warnings.warn` function
    Examples
    --------
    The effects are demonstrated by this example::
        >>> from dynamo.docrep import safe_modulo
        >>> s = "That's %(one)s string %(with)s missing 'with' and %s key"
        >>> s % {'one': 1}          # raises KeyError because of missing 'with'
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        KeyError: 'with'
        >>> s % {'one': 1, 'with': 2}        # raises TypeError because of '%s'
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: not enough arguments for format string
        >>> safe_modulo(s, {'one': 1})
        "That's 1 string %(with)s missing 'with' and %s key"
    """
    try:
        return s % meta
    except (ValueError, TypeError, KeyError):
        # replace the missing fields by %%
        keys = substitution_pattern.finditer(s)
        for m in keys:
            key = m.group("key")
            if not isinstance(meta, dict) or key not in meta:
                if print_warning:
                    warn("%r is not a valid key!" % key, SyntaxWarning, stacklevel)
                full = m.group()
                s = s.replace(full, "%" + full)
        if "KEY" not in checked:
            return safe_modulo(
                s,
                meta,
                checked=checked + "KEY",
                print_warning=print_warning,
                stacklevel=stacklevel,
            )
        if not isinstance(meta, dict) or "VALUE" in checked:
            raise
        s = re.sub(
            r"""(?<!%)(%%)*%(?!%) # uneven number of %
                    \s*(\w|$)         # format strings""",
            "%\g<0>",
            s,
            flags=re.VERBOSE,
        )
        return safe_modulo(
            s,
            meta,
            checked=checked + "VALUE",
            print_warning=print_warning,
            stacklevel=stacklevel,
        )


class DocstringProcessor(object):
    """Class that is intended to process docstrings
    It is, but only to minor extends, inspired by the
    :class:`matplotlib.docstring.Substitution` class.
    Examples
    --------
    Create docstring processor via::
        >>> from dynamo.docrep import DocstringProcessor
        >>> d = DocstringProcessor(doc_key='My doc string')
    And then use it as a decorator to process the docstring::
        >>> @d
        ... def doc_test():
        ...     '''That's %(doc_key)s'''
        ...     pass
        >>> print(doc_test.__doc__)
        That's My doc string
    Use the :meth:`get_sectionsf` method to extract Parameter sections (or
    others) form the docstring for later usage (and make sure, that the
    docstring is dedented)::
        >>> @d.get_sectionsf('docstring_example',
        ...                  sections=['Parameters', 'Examples'])
        ... @d.dedent
        ... def doc_test(a=1, b=2):
        ...     '''
        ...     That's %(doc_key)s
        ...
        ...     Parameters
        ...     ----------
        ...     a: int, optional
        ...         A dummy parameter description
        ...     b: int, optional
        ...         A second dummy parameter
        ...
        ...     Examples
        ...     --------
        ...     Some dummy example doc'''
        ...     print(a)
        >>> @d.dedent
        ... def second_test(a=1, b=2):
        ...     '''
        ...     My second function where I want to use the docstring from
        ...     above
        ...
        ...     Parameters
        ...     ----------
        ...     %(docstring_example.parameters)s
        ...
        ...     Examples
        ...     --------
        ...     %(docstring_example.examples)s'''
        ...     pass
        >>> print(second_test.__doc__)
        My second function where I want to use the docstring from
        above
        <BLANKLINE>
        Parameters
        ----------
        a: int, optional
            A dummy parameter description
        b: int, optional
            A second dummy parameter
        <BLANKLINE>
        Examples
        --------
        Some dummy example doc
    Another example uses non-dedented docstrings::
        >>> @d.get_sectionsf('not_dedented')
        ... def doc_test2(a=1):
        ...     '''That's the summary
        ...
        ...     Parameters
        ...     ----------
        ...     a: int, optional
        ...         A dummy parameter description'''
        ...     print(a)
    These sections must then be used with the :meth:`with_indent` method to
    indent the inserted parameters::
        >>> @d.with_indent(4)
        ... def second_test2(a=1):
        ...     '''
        ...     My second function where I want to use the docstring from
        ...     above
        ...
        ...     Parameters
        ...     ----------
        ...     %(not_dedented.parameters)s'''
        ...     pass
    """

    #: :class:`dict`. Dictionary containing the compiled patterns to identify
    #: the Parameters, Other Parameters, Warnings and Notes sections in a
    #: docstring
    patterns = {}

    #: :class:`dict`. Dictionary containing the parameters that are used in for
    #: substitution.
    params = {}

    #: sections that behave the same as the `Parameter` section by defining a
    #: list
    param_like_sections = ["Parameters", "Other Parameters", "Returns", "Raises"]
    #: sections that include (possibly not list-like) text
    text_sections = ["Warnings", "Notes", "Examples", "See Also", "References"]

    #: The action on how to react on classes in python 2
    #:
    #: When calling::
    #:
    #:     >>> @docstrings
    #:     ... class NewClass(object):
    #:     ...     """%(replacement)s"""
    #:
    #: This normaly raises an AttributeError, because the ``__doc__`` attribute
    #: of a class in python 2 is not writable. This attribute may be one of
    #: ``'ignore', 'raise' or 'warn'``
    python2_classes = "ignore"

    def __init__(self, *args, **kwargs):
        """
    Parameters
    ----------
    ``*args`` and ``**kwargs``
        Parameters that shall be used for the substitution. Note that you can
        only provide either ``*args`` or ``**kwargs``, furthermore most of the
        methods like `get_sectionsf` require ``**kwargs`` to be provided."""
        if len(args) and len(kwargs):
            raise ValueError("Only positional or keyword args are allowed")
        self.params = args or kwargs
        patterns = {}
        all_sections = self.param_like_sections + self.text_sections
        for section in self.param_like_sections:
            patterns[section] = re.compile(
                "(?s)(?<=%s\n%s\n)(.+?)(?=\n\n\S+|$)" % (section, "-" * len(section))
            )
        all_sections_patt = "|".join(
            "%s\n%s\n" % (s, "-" * len(s)) for s in all_sections
        )
        # examples and see also
        for section in self.text_sections:
            patterns[section] = re.compile(
                "(?s)(?<=%s\n%s\n)(.+?)(?=%s|$)"
                % (section, "-" * len(section), all_sections_patt)
            )
        self._extended_summary_patt = re.compile(
            "(?s)(.+?)(?=%s|$)" % all_sections_patt
        )
        self._all_sections_patt = re.compile(all_sections_patt)
        self.patterns = patterns

    def __call__(self, func):
        """
        Substitute in a docstring of a function with :attr:`params`
        Parameters
        ----------
        func: function
            function with the documentation whose sections
            shall be inserted from the :attr:`params` attribute
        See Also
        --------
        dedent: also dedents the doc
        with_indent: also indents the doc"""
        doc = func.__doc__ and safe_modulo(func.__doc__, self.params, stacklevel=3)
        return self._set_object_doc(func, doc)

    def get_sections(self, s, base, sections=["Parameters", "Other Parameters"]):
        """
        Method that extracts the specified sections out of the given string if
        (and only if) the docstring follows the numpy documentation guidelines
        [1]_. Note that the section either must appear in the
        :attr:`param_like_sections` or the :attr:`text_sections` attribute.
        Parameters
        ----------
        s: str
            Docstring to split
        base: str
            base to use in the :attr:`sections` attribute
        sections: list of str
            sections to look for. Each section must be followed by a newline
            character ('\\n') and a bar of '-' (following the numpy (napoleon)
            docstring conventions).
        Returns
        -------
        str
            The replaced string
        References
        ----------
        .. [1] https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
        See Also
        --------
        delete_params, keep_params, delete_types, keep_types, delete_kwargs:
            For manipulating the docstring sections
        save_docstring:
            for saving an entire docstring
        """
        params = self.params
        # Remove the summary and dedent the rest
        s = self._remove_summary(s)
        for section in sections:
            key = "%s.%s" % (base, section.lower().replace(" ", "_"))
            params[key] = self._get_section(s, section)
        return s

    def _remove_summary(self, s):
        # if the string does not start with one of the sections, we remove the
        # summary
        if not self._all_sections_patt.match(s.lstrip()):
            # remove the summary
            lines = summary_patt.sub("", s, 1).splitlines()
            # look for the first line with content
            first = next((i for i, l in enumerate(lines) if l.strip()), 0)
            # dedent the lines
            s = inspect.cleandoc("\n" + "\n".join(lines[first:]))
        return s

    def _get_section(self, s, section):
        try:
            return self.patterns[section].search(s).group(0).rstrip()
        except AttributeError:
            return ""

    def get_sectionsf(self, *args, **kwargs):
        """
        Decorator method to extract sections from a function docstring
        Parameters
        ----------
        ``*args`` and ``**kwargs``
            See the :meth:`get_sections` method. Note, that the first argument
            will be the docstring of the specified function
        Returns
        -------
        function
            Wrapper that takes a function as input and registers its sections
            via the :meth:`get_sections` method"""

        def func(f):
            doc = f.__doc__
            self.get_sections(doc or "", *args, **kwargs)
            return f

        return func

    def _set_object_doc(self, obj, doc, stacklevel=3):
        """Convenience method to set the __doc__ attribute of a python object
        """
        if isinstance(obj, types.MethodType) and six.PY2:
            obj = obj.im_func
        try:
            obj.__doc__ = doc
        except AttributeError:  # probably python2 class
            if self.python2_classes != "raise" and (inspect.isclass(obj) and six.PY2):
                if self.python2_classes == "warn":
                    warn(
                        "Cannot modify docstring of classes in python2!",
                        stacklevel=stacklevel,
                    )
            else:
                raise
        return obj

    def dedent(self, func):
        """
        Dedent the docstring of a function and substitute with :attr:`params`
        Parameters
        ----------
        func: function
            function with the documentation to dedent and whose sections
            shall be inserted from the :attr:`params` attribute"""
        doc = func.__doc__ and self.dedents(func.__doc__, stacklevel=4)
        return self._set_object_doc(func, doc)

    def dedents(self, s, stacklevel=3):
        """
        Dedent a string and substitute with the :attr:`params` attribute
        Parameters
        ----------
        s: str
            string to dedent and insert the sections of the :attr:`params`
            attribute
        stacklevel: int
            The stacklevel for the warning raised in :func:`safe_module` when
            encountering an invalid key in the string"""
        s = inspect.cleandoc(s)
        return safe_modulo(s, self.params, stacklevel=stacklevel)

    def with_indent(self, indent=0):
        """
        Substitute in the docstring of a function with indented :attr:`params`
        Parameters
        ----------
        indent: int
            The number of spaces that the substitution should be indented
        Returns
        -------
        function
            Wrapper that takes a function as input and substitutes it's
            ``__doc__`` with the indented versions of :attr:`params`
        See Also
        --------
        with_indents, dedent"""

        def replace(func):
            doc = func.__doc__ and self.with_indents(
                func.__doc__, indent=indent, stacklevel=4
            )
            return self._set_object_doc(func, doc)

        return replace

    def with_indents(self, s, indent=0, stacklevel=3):
        """
        Substitute a string with the indented :attr:`params`
        Parameters
        ----------
        s: str
            The string in which to substitute
        indent: int
            The number of spaces that the substitution should be indented
        stacklevel: int
            The stacklevel for the warning raised in :func:`safe_module` when
            encountering an invalid key in the string
        Returns
        -------
        str
            The substituted string
        See Also
        --------
        with_indent, dedents"""
        # we make a new dictionary with objects that indent the original
        # strings if necessary. Note that the first line is not indented
        d = {
            key: _StrWithIndentation(val, indent)
            for key, val in six.iteritems(self.params)
        }
        return safe_modulo(s, d, stacklevel=stacklevel)

    def delete_params(self, base_key, *params):
        """
        Method to delete a parameter from a parameter documentation.
        This method deletes the given `param` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation without the description of the param. This method works
        for the ``'Parameters'`` sections.
        The new docstring without the selected parts will be accessible as
        ``base_key + '.no_' + '|'.join(params)``, e.g.
        ``'original_key.no_param1|param2'``.
        See the :meth:`keep_params` method for an example.
        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        ``*params``
            str. Parameter identifier of which the documentations shall be
            deleted
        See Also
        --------
        delete_types, keep_params"""
        self.params[base_key + ".no_" + "|".join(params)] = self.delete_params_s(
            self.params[base_key], params
        )

    @staticmethod
    def delete_params_s(s, params):
        """
        Delete the given parameters from a string
        Same as :meth:`delete_params` but does not use the :attr:`params`
        dictionary
        Parameters
        ----------
        s: str
            The string of the parameters section
        params: list of str
            The names of the parameters to delete
        Returns
        -------
        str
            The modified string `s` without the descriptions of `params`
        """
        patt = "(?s)" + "|".join("(?<=\n)" + s + "\s*:.+?\n(?=\S+|$)" for s in params)
        return re.sub(patt, "", "\n" + s.strip() + "\n").strip()

    def delete_kwargs(self, base_key, args=None, kwargs=None):
        """
        Deletes the ``*args`` or ``**kwargs`` part from the parameters section
        Either `args` or `kwargs` must not be None. The resulting key will be
        stored in
        ``base_key + 'no_args'``
            if `args` is not None and `kwargs` is None
        ``base_key + 'no_kwargs'``
            if `args` is None and `kwargs` is not None
        ``base_key + 'no_args_kwargs'``
            if `args` is not None and `kwargs` is not None
        Parameters
        ----------
        base_key: str
            The key in the :attr:`params` attribute to use
        args: None or str
            The string for the args to delete
        kwargs: None or str
            The string for the kwargs to delete
        Notes
        -----
        The type name of `args` in the base has to be like ````*<args>````
        (i.e. the `args` argument preceeded by a ``'*'`` and enclosed by double
        ``'`'``). Similarily, the type name of `kwargs` in `s` has to be like
        ````**<kwargs>````"""
        if not args and not kwargs:
            warn("Neither args nor kwargs are given. I do nothing for %s" % (base_key))
            return
        ext = ".no" + ("_args" if args else "") + ("_kwargs" if kwargs else "")
        self.params[base_key + ext] = self.delete_kwargs_s(
            self.params[base_key], args, kwargs
        )

    @classmethod
    def delete_kwargs_s(cls, s, args=None, kwargs=None):
        """
        Deletes the ``*args`` or ``**kwargs`` part from the parameters section
        Either `args` or `kwargs` must not be None.
        Parameters
        ----------
        s: str
            The string to delete the args and kwargs from
        args: None or str
            The string for the args to delete
        kwargs: None or str
            The string for the kwargs to delete
        Notes
        -----
        The type name of `args` in `s` has to be like ````*<args>```` (i.e. the
        `args` argument preceeded by a ``'*'`` and enclosed by double ``'`'``).
        Similarily, the type name of `kwargs` in `s` has to be like
        ````**<kwargs>````"""
        if not args and not kwargs:
            return s
        types = []
        if args is not None:
            types.append("`?`?\*%s`?`?" % args)
        if kwargs is not None:
            types.append("`?`?\*\*%s`?`?" % kwargs)
        return cls.delete_types_s(s, types)

    def delete_types(self, base_key, out_key, *types):
        """
        Method to delete a parameter from a parameter documentation.
        This method deletes the given `param` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation without the description of the param. This method works
        for ``'Results'`` like sections.
        See the :meth:`keep_types` method for an example.
        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        out_key: str
            Extension for the base key (the final key will be like
            ``'%s.%s' % (base_key, out_key)``
        ``*types``
            str. The type identifier of which the documentations shall deleted
        See Also
        --------
        delete_params"""
        self.params["%s.%s" % (base_key, out_key)] = self.delete_types_s(
            self.params[base_key], types
        )

    @staticmethod
    def delete_types_s(s, types):
        """
        Delete the given types from a string
        Same as :meth:`delete_types` but does not use the :attr:`params`
        dictionary
        Parameters
        ----------
        s: str
            The string of the returns like section
        types: list of str
            The type identifiers to delete
        Returns
        -------
        str
            The modified string `s` without the descriptions of `types`
        """
        patt = "(?s)" + "|".join("(?<=\n)" + s + "\n.+?\n(?=\S+|$)" for s in types)
        return re.sub(patt, "", "\n" + s.strip() + "\n",).strip()

    def keep_params(self, base_key, *params):
        """
        Method to keep only specific parameters from a parameter documentation.
        This method extracts the given `param` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation with only the description of the param. This method works
        for ``'Parameters'`` like sections.
        The new docstring with the selected parts will be accessible as
        ``base_key + '.' + '|'.join(params)``, e.g.
        ``'original_key.param1|param2'``
        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        ``*params``
            str. Parameter identifier of which the documentations shall be
            in the new section
        See Also
        --------
        keep_types, delete_params
        Examples
        --------
        To extract just two parameters from a function and reuse their
        docstrings, you can type::
            >>> from dynamo.docrep import DocstringProcessor
            >>> d = DocstringProcessor()
            >>> @d.get_sectionsf('do_something')
            ... def do_something(a=1, b=2, c=3):
            ...     '''
            ...     That's %(doc_key)s
            ...
            ...     Parameters
            ...     ----------
            ...     a: int, optional
            ...         A dummy parameter description
            ...     b: int, optional
            ...         A second dummy parameter that will be excluded
            ...     c: float, optional
            ...         A third parameter'''
            ...     print(a)
            >>> d.keep_params('do_something.parameters', 'a', 'c')
            >>> @d.dedent
            ... def do_less(a=1, c=4):
            ...     '''
            ...     My second function with only `a` and `c`
            ...
            ...     Parameters
            ...     ----------
            ...     %(do_something.parameters.a|c)s'''
            ...     pass
            >>> print(do_less.__doc__)
            My second function with only `a` and `c`
            <BLANKLINE>
            Parameters
            ----------
            a: int, optional
                A dummy parameter description
            c: float, optional
                A third parameter
        Equivalently, you can use the :meth:`delete_params` method to remove
        parameters::
            >>> d.delete_params('do_something.parameters', 'b')
            >>> @d.dedent
            ... def do_less(a=1, c=4):
            ...     '''
            ...     My second function with only `a` and `c`
            ...
            ...     Parameters
            ...     ----------
            ...     %(do_something.parameters.no_b)s'''
            ...     pass
        """
        self.params[base_key + "." + "|".join(params)] = self.keep_params_s(
            self.params[base_key], params
        )

    @staticmethod
    def keep_params_s(s, params):
        """
        Keep the given parameters from a string
        Same as :meth:`keep_params` but does not use the :attr:`params`
        dictionary
        Parameters
        ----------
        s: str
            The string of the parameters like section
        params: list of str
            The parameter names to keep
        Returns
        -------
        str
            The modified string `s` with only the descriptions of `params`
        """
        patt = "(?s)" + "|".join("(?<=\n)" + s + "\s*:.+?\n(?=\S+|$)" for s in params)
        return "".join(re.findall(patt, "\n" + s.strip() + "\n")).rstrip()

    def keep_types(self, base_key, out_key, *types):
        """
        Method to keep only specific parameters from a parameter documentation.
        This method extracts the given `type` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation with only the description of the type. This method works
        for the ``'Results'`` sections.
        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        out_key: str
            Extension for the base key (the final key will be like
            ``'%s.%s' % (base_key, out_key)``
        ``*types``
            str. The type identifier of which the documentations shall be
            in the new section
        See Also
        --------
        delete_types, keep_params
        Examples
        --------
        To extract just two return arguments from a function and reuse their
        docstrings, you can type::
            >>> from dynamo.docrep import DocstringProcessor
            >>> d = DocstringProcessor()
            >>> @d.get_sectionsf('do_something', sections=['Returns'])
            ... def do_something():
            ...     '''
            ...     That's %(doc_key)s
            ...
            ...     Returns
            ...     -------
            ...     float
            ...         A random number
            ...     int
            ...         A random integer'''
            ...     return 1.0, 4
            >>> d.keep_types('do_something.returns', 'int_only', 'int')
            >>> @d.dedent
            ... def do_less():
            ...     '''
            ...     My second function that only returns an integer
            ...
            ...     Returns
            ...     -------
            ...     %(do_something.returns.int_only)s'''
            ...     return do_something()[1]
            >>> print(do_less.__doc__)
            My second function that only returns an integer
            <BLANKLINE>
            Returns
            -------
            int
                A random integer
        Equivalently, you can use the :meth:`delete_types` method to remove
        parameters::
            >>> d.delete_types('do_something.returns', 'no_float', 'float')
            >>> @d.dedent
            ... def do_less():
            ...     '''
            ...     My second function with only `a` and `c`
            ...
            ...     Returns
            ...     ----------
            ...     %(do_something.returns.no_float)s'''
            ...     return do_something()[1]
        """
        self.params["%s.%s" % (base_key, out_key)] = self.keep_types_s(
            self.params[base_key], types
        )

    @staticmethod
    def keep_types_s(s, types):
        """
        Keep the given types from a string
        Same as :meth:`keep_types` but does not use the :attr:`params`
        dictionary
        Parameters
        ----------
        s: str
            The string of the returns like section
        types: list of str
            The type identifiers to keep
        Returns
        -------
        str
            The modified string `s` with only the descriptions of `types`
        """
        patt = "(?s)" + "|".join("(?<=\n)" + s + "\n.+?\n(?=\S+|$)" for s in types)
        return "".join(re.findall(patt, "\n" + s.strip() + "\n")).rstrip()

    def save_docstring(self, key):
        """
        Descriptor method to save_fig a docstring from a function
        Like the :meth:`get_sectionsf` method this method serves as a
        descriptor for functions but saves the entire docstring"""

        def func(f):
            self.params[key] = f.__doc__ or ""
            return f

        return func

    def get_summary(self, s, base=None):
        """
        Get the summary of the given docstring
        This method extracts the summary from the given docstring `s` which is
        basicly the part until two newlines appear
        Parameters
        ----------
        s: str
            The docstring to use
        base: str or None
            A key under which the summary shall be stored in the :attr:`params`
            attribute. If not None, the summary will be stored in
            ``base + '.summary'``. Otherwise, it will not be stored at all
        Returns
        -------
        str
            The extracted summary"""
        summary = summary_patt.search(s).group()
        if base is not None:
            self.params[base + ".summary"] = summary
        return summary

    def get_summaryf(self, *args, **kwargs):
        """
        Extract the summary from a function docstring
        Parameters
        ----------
        ``*args`` and ``**kwargs``
            See the :meth:`get_summary` method. Note, that the first argument
            will be the docstring of the specified function
        Returns
        -------
        function
            Wrapper that takes a function as input and registers its summary
            via the :meth:`get_summary` method"""

        def func(f):
            doc = f.__doc__
            self.get_summary(doc or "", *args, **kwargs)
            return f

        return func

    def get_extended_summary(self, s, base=None):
        """Get the extended summary from a docstring
        This here is the extended summary
        Parameters
        ----------
        s: str
            The docstring to use
        base: str or None
            A key under which the summary shall be stored in the :attr:`params`
            attribute. If not None, the summary will be stored in
            ``base + '.summary_ext'``. Otherwise, it will not be stored at
            all
        Returns
        -------
        str
            The extracted extended summary"""
        # Remove the summary and dedent
        s = self._remove_summary(s)
        ret = ""
        if not self._all_sections_patt.match(s):
            m = self._extended_summary_patt.match(s)
            if m is not None:
                ret = m.group().strip()
        if base is not None:
            self.params[base + ".summary_ext"] = ret
        return ret

    def get_extended_summaryf(self, *args, **kwargs):
        """Extract the extended summary from a function docstring
        This function can be used as a decorator to extract the extended
        summary of a function docstring (similar to :meth:`get_sectionsf`).
        Parameters
        ----------
        ``*args`` and ``**kwargs``
            See the :meth:`get_extended_summary` method. Note, that the first
            argument will be the docstring of the specified function
        Returns
        -------
        function
            Wrapper that takes a function as input and registers its summary
            via the :meth:`get_extended_summary` method"""

        def func(f):
            doc = f.__doc__
            self.get_extended_summary(doc or "", *args, **kwargs)
            return f

        return func

    def get_full_description(self, s, base=None):
        """Get the full description from a docstring
        This here and the line above is the full description (i.e. the
        combination of the :meth:`get_summary` and the
        :meth:`get_extended_summary`) output
        Parameters
        ----------
        s: str
            The docstring to use
        base: str or None
            A key under which the description shall be stored in the
            :attr:`params` attribute. If not None, the summary will be stored
            in ``base + '.full_desc'``. Otherwise, it will not be stored
            at all
        Returns
        -------
        str
            The extracted full description"""
        summary = self.get_summary(s)
        extended_summary = self.get_extended_summary(s)
        ret = (summary + "\n\n" + extended_summary).strip()
        if base is not None:
            self.params[base + ".full_desc"] = ret
        return ret

    def get_full_descriptionf(self, *args, **kwargs):
        """Extract the full description from a function docstring
        This function can be used as a decorator to extract the full
        descriptions of a function docstring (similar to
        :meth:`get_sectionsf`).
        Parameters
        ----------
        ``*args`` and ``**kwargs``
            See the :meth:`get_full_description` method. Note, that the first
            argument will be the docstring of the specified function
        Returns
        -------
        function
            Wrapper that takes a function as input and registers its summary
            via the :meth:`get_full_description` method"""

        def func(f):
            doc = f.__doc__
            self.get_full_description(doc or "", *args, **kwargs)
            return f

        return func
