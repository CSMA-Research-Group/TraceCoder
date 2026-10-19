"""Check that instrumentation only inserts observational print statements.

AST preservation and unchanged test outcomes are safeguards, not a formal proof
of observational purity for Python objects with overloaded operations.
"""
import ast
import copy


def _probe(node):
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    call = node.value
    if not (isinstance(call.func, ast.Name) and call.func.id == "print") or call.keywords:
        return False
    allowed = (ast.Constant, ast.Name, ast.Load, ast.JoinedStr, ast.FormattedValue,
               ast.Attribute, ast.Subscript, ast.Slice, ast.Tuple, ast.List,
               ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.Mod, ast.USub, ast.UAdd, ast.Not, ast.Compare, ast.Eq,
               ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)
    return all(isinstance(part, allowed) and not (
        isinstance(part, ast.Attribute) and part.attr.startswith("__"))
        for arg in call.args for part in ast.walk(arg))


def validate_instrumentation(original, instrumented):
    left, right = ast.parse(original), ast.parse(instrumented)
    # A shadowed print could execute arbitrary logic in place of the probe.
    for node in ast.walk(left):
        if ((isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "print")
                or (isinstance(node, ast.arg) and node.arg == "print")
                or (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == "print")
                or (isinstance(node, ast.alias) and (node.asname or node.name) == "print")):
            raise ValueError("不能在 print 被重定义的程序上验证打印插桩。")
    inserted = []

    def equal(a, b):
        if isinstance(a, ast.AST):
            if type(a) is not type(b):
                return False
            return all(equal(v, getattr(b, field)) for field, v in ast.iter_fields(a))
        if isinstance(a, list):
            if not isinstance(b, list):
                return False
            if all(isinstance(v, ast.stmt) for v in a + b):
                i = 0
                for candidate in b:
                    checkpoint = len(inserted)
                    if i < len(a) and equal(a[i], candidate):
                        i += 1
                    else:
                        del inserted[checkpoint:]
                        if not _probe(candidate):
                            return False
                        inserted.append(candidate.lineno)
                return i == len(a)
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    if not equal(left, right):
        raise ValueError("插桩改变了原程序 AST，或新增了带调用/副作用的探针。")
    if not inserted:
        raise ValueError("插桩响应没有新增打印探针。")
    return {"ast_preserved": True, "probe_count": len(inserted), "probe_lines": sorted(inserted)}


def fixture_probes(source):
    """Deterministic instrumentation used only by the labelled teaching fixture."""
    class Probes(ast.NodeTransformer):
        def print_node(self, label, values):
            return ast.Expr(value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Constant(value=label)] + [copy.deepcopy(v) for v in values], keywords=[]))

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            args = [ast.Name(id=a.arg, ctx=ast.Load()) for a in node.args.args if a.arg != "self"]
            start = 1 if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str) else 0
            node.body.insert(start, self.print_node("[enter " + node.name + "]", args))
            return node

        def visit_Assign(self, node):
            values = []
            for target in node.targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    value = copy.deepcopy(target)
                    value.ctx = ast.Load()
                    values.append(value)
            return [node, self.print_node("[L%d assigned]" % node.lineno, values)] if values else node

        def visit_AugAssign(self, node):
            if isinstance(node.target, (ast.Name, ast.Attribute)):
                value = copy.deepcopy(node.target)
                value.ctx = ast.Load()
                return [node, self.print_node("[L%d updated]" % node.lineno, [value])]
            return node

        def visit_Return(self, node):
            values = [node.value] if isinstance(node.value, (ast.Name, ast.Constant)) else []
            return [self.print_node("[L%d return]" % node.lineno, values), node]

    tree = Probes().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree)
    validate_instrumentation(source, result)
    return result
