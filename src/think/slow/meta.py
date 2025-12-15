#!/usr/bin/env python3

""" Metaprogramming. """

__all__ = [
    'build',
]

import os
import inspect

indent = ' '*4

LIFTED_NAMES = {
    'a': 'obj1',
    'b': 'obj2',
    't': 'obj',
    'v': 'value',
    'ts': 'objs',
}

def get_promotions(jit_func):
    triples = []
    num_plurals = 0
    for name in inspect.signature(jit_func).parameters:
        if len(name) == 1:
            promotion = 'slow.to_vector'
        elif len(name) == 2:
            promotion = 'slow.to_array'
            num_plurals += 1
        elif len(name) > 2:
            promotion = None
        else:
            raise ValueError(jit_func)

        if num_plurals >= 2:
            raise NotImplementedError(
                f"JIT function with >= 2 plurals encountered. "
                f"Implement this in meta.py."
            )

        old_name = name
        new_name = LIFTED_NAMES.get(name, name)
        triple = (old_name, new_name, promotion)
        triples.append(triple)

    return triples


def to_polymorphic_code(jit_func):

    triples = get_promotions(jit_func)

    old_names, new_names, promotions = zip(*triples)
    body = []
    for old_name, new_name, promotion in triples:
        if old_name != new_name:
            assert promotion.startswith('slow')
            line = f"{old_name} = {promotion}({new_name})"
            body.append(line)
        else:
            assert promotion is None

    name = jit_func.__name__
    new_args = ', '.join(new_names)
    old_args = ', '.join(old_names)
    head = f"def {name}({new_args}):"
    tail = f"return fast.{name}({old_args})"
    code = f"\n{indent}".join([head, *body, tail])
    return code

def to_polymorphic_function(jit_func):
    code = to_polymorphic_code(jit_func)
    namespace = {}
    exec(code, globals(), namespace)
    return namespace[name]


def regenerate_module(fast, output_path):

    exports = []
    functions = []

    for name in fast.__all__:
        jit_func = getattr(fast, name)
        exports.append(f"{name!r}")
        code = to_polymorphic_code(jit_func)
        functions.append(code)

    import textwrap

    imports = textwrap.dedent(f"""
    #!/usr/bin/env python3

    from think import fast
    from think import slow
    """)

    exports = f",\n{indent}".join(exports)

    functions = f"\n\n".join(functions)

    module = textwrap.dedent("""
    {imports}

    __all__ = [
        {exports}
    ]

    {functions}
    """).strip().format(
        imports=imports.strip(),
        exports=exports.strip(),
        functions=functions.strip(),
    )
    module = f"{module}\n"

    with open(output_path, 'w') as fp:
        fp.write(module)

    return module

def modification_time(path):
    if os.path.exists(path):
        return os.stat(path).st_mtime
    else:
        return 0

def build(force=False):
    from think import fast
    from think import slow
    dirname = os.path.dirname(slow.__file__)
    fast_path = fast.__file__
    slow_path = os.path.join(dirname, 'poly.py')
    last_path = os.path.join(dirname, '.lastbuild')
    if force:
        os.system(f"touch {last_path!r}")
        return regenerate_module(fast, slow_path)
    if not os.path.exists(slow_path):
        os.system(f"touch {last_path!r}")
        return regenerate_module(fast, slow_path)
    fast_mtime = modification_time(fast_path)
    slow_mtime = modification_time(slow_path)
    last_mtime = modification_time(last_path)
    if (last_mtime < fast_mtime) or (last_mtime < slow_mtime):
        os.system(f"touch {last_path!r}")
        return regenerate_module(fast, slow_path)
    return "Nothing needed"

if __name__ == '__main__':
    module = build(force=True)
    print(module)

