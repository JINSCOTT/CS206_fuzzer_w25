import ast
import torch

class TorchMathTransformer(ast.NodeTransformer):
    def _ensure_tensor(self, node):
        """Helper function to wrap a constant in torch.tensor if needed."""
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # Wrap the number in torch.tensor()
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='tensor',
                    ctx=ast.Load()
                ),
                args=[node],  # Pass the original constant node as the argument
                keywords=[]
            )
            return ast.copy_location(new_node, node)  # Preserve location info
        """
        return node

    def visit_FunctionDef(self, node):
        """Visit function definitions to modify the 'forward' function."""
        if node.name == 'forward':
            # Remove the unnecessary tensor check
            node.body = node.body  # Leave the function body as is

        self.generic_visit(node)  # Visit the rest of the function
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)  # process children first
        # Ensure both operands are tensors before applying torch operations
        node.left = self._ensure_tensor(node.left)
        node.right = self._ensure_tensor(node.right)

        # Replace addition (a + b) with torch.add(a, b)
        if isinstance(node.op, ast.Add):
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='add',
                    ctx=ast.Load()
                ),
                args=[node.left, node.right],
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        
        # Replace subtraction (a - b) with torch.sub(a, b)
        elif isinstance(node.op, ast.Sub):
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='sub',
                    ctx=ast.Load()
                ),
                args=[node.left, node.right],
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        
        # Replace multiplication (a * b) with torch.mul(a, b)
        elif isinstance(node.op, ast.Mult):
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='mul',
                    ctx=ast.Load()
                ),
                args=[node.left, node.right],
                keywords=[]
            )
            return ast.copy_location(new_node, node)

        # Replace division (a / b) with torch.div(a, b)
        elif isinstance(node.op, ast.Div):
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='div',
                    ctx=ast.Load()
                ),
                args=[node.left, node.right],
                keywords=[]
            )
            return ast.copy_location(new_node, node)

        # Replace matrix multiplication (a @ b) with torch.matmul(a, b)
        elif isinstance(node.op, ast.MatMult):
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='matmul',
                    ctx=ast.Load()
            ),
                args=[node.left, node.right],
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        elif isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and node.right.value == 0.5:
                # create sqrt replace
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='sqrt',
                    ctx=ast.Load()
                ),
                args=[node.left],
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        # Replace exponentiation (a ** b) with torch.pow(a, b)
        elif isinstance(node.op, ast.Pow):
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='pow',
                    ctx=ast.Load()
                ),
                args=[node.left, node.right],
                keywords=[]
            )
            return ast.copy_location(new_node, node)

        else:
            return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        """
        # Ensure both operands are tensors before comparison
        node.left = self._ensure_tensor(node.left)
        node.comparators = [self._ensure_tensor(comp) for comp in node.comparators]
        
        if len(node.ops) == 1:  # Handle simple comparisons
            left = node.left
            right = node.comparators[0]  # Get the right side of the comparison
            op = node.ops[0]
            if (isinstance(left,ast.Attribute) and (left.attr =="dim")):
                new_node = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr="eq",
                        ctx=ast.Load()
                    ),
                    args=[left,right],
                    keywords=[]
                )
                return new_node
            # Heuristic: If either operand is a string literal, don't transform
            if (isinstance(left, ast.Constant) and isinstance(left.value, str)) or \
               (isinstance(right, ast.Constant) and isinstance(right.value, str)) or\
                (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)):
                return node  # Don't transform

            torch_compare_map = {
                ast.Eq: 'eq',
                ast.NotEq: 'ne',
                ast.Lt: 'lt',
                ast.LtE: 'le',
                ast.Gt: 'gt',
                ast.GtE: 'ge'
            }

            if type(op) in torch_compare_map:
                torch_op = torch_compare_map[type(op)]
                new_node = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr=torch_op,
                        ctx=ast.Load()
                    ),
                    args=[left, right],
                    keywords=[]
                )
                return ast.copy_location(new_node, node)
                """
        return node  # Leave chained comparisons unchanged for now

    def visit_UnaryOp(self, node):
        self.generic_visit(node)  # Visit any child nodes first
        if isinstance(node.op, ast.USub):
            # Handle unary minus (-x)
            # Apply _ensure_tensor to the operand FIRST, then negate
            if isinstance(node.operand, ast.Name):
                new_node = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='torch', ctx=ast.Load()),
                        attr='neg',
                        ctx=ast.Load()
                    ),
                    args=[node.operand],
                    keywords=[]
                )
                return ast.copy_location(new_node, node)
        elif isinstance(node.op, ast.Not):
            # Handle logical not (not x)
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='logical_not',
                    ctx=ast.Load()
                ),
                args=[node.operand],
                keywords=[]
            )
            return ast.copy_location(new_node, node)

        elif isinstance(node.op, ast.Invert):
            # Handle bitwise not (~x)
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='bitwise_not',
                    ctx=ast.Load()
                ),
                args=[node.operand],
                keywords=[]
            )
            return ast.copy_location(new_node, node)

        return node  # Leave other unary ops unchanged

    def visit_Call(self, node):
        self.generic_visit(node)
        # Replace built-in pow(x, y) with torch.pow(x, y)
        if isinstance(node.func, ast.Name) and node.func.id == "pow":
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='pow',
                    ctx=ast.Load()
                ),
                args=node.args,
                keywords=[]
            )
            return ast.copy_location(new_node, node)

        # Replace built-in abs(x) with torch.abs(x)
        if isinstance(node.func, ast.Name) and node.func.id == "abs":
            new_node = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='torch', ctx=ast.Load()),
                    attr='abs',
                    ctx=ast.Load()
                ),
                args=node.args,
                keywords=[]
            )
            return ast.copy_location(new_node, node)
        # Handle x.transpose(-1, -2) correctly
        if (isinstance(node.func, ast.Attribute) and
            node.func.attr == 'transpose' and
            isinstance(node.func.value, ast.Name) and
            len(node.args) == 2):  # Basic check for transpose(dim1, dim2)
            
            # Ensure the transpose dimensions are tensors:
            node.args[0] = self._ensure_tensor(node.args[0])
            node.args[1] = self._ensure_tensor(node.args[1])
            return node  # Return the modified node
        
        if (isinstance(node.func, ast.Attribute) and
            node.func.attr == 'eq' and
            isinstance(node.func.value, ast.Name)):
            
            return node.args[0]

        return node
    def visit_IfExp(self, node):
        self.generic_visit(node)

        # Transform the test condition using torch.eq if it's a comparison involving dim
        if isinstance(node.test, ast.Compare):
            if len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq):
                left = node.test.left
                right = node.test.comparators[0]
                if (isinstance(left,ast.Attribute) and (left.attr =="dim")):
                     new_node = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id='torch', ctx=ast.Load()),
                            attr="eq",
                            ctx=ast.Load()
                        ),
                        args=[left,self._ensure_tensor(right)],
                        keywords=[]
                    )
                     node.test = new_node
                     return node


        return node
    def visit_List(self, node):
        self.generic_visit(node)
        return node
    def visit_Module(self, node):
        self.generic_visit(node)
        return node
    def visit_Assign(self, node):
        self.generic_visit(node)
        return node


if __name__ == "__main__":
    # Example usage
    
    code = """
import torch
import torch.nn as nn

class MathOperationsModule(nn.Module):
    def __init__(self):
        super(MathOperationsModule, self).__init__()

    def forward(self, x):
        # Perform various mathematical operations
        x = x * 2.5                      # Scalar multiplication
        x = x - torch.mean(x)            # Subtract mean
        x = torch.exp(x)                 # Exponential
        x = x / (torch.abs(x) + 1)       # Normalize with abs
        x = torch.matmul(x, x.transpose(-1, -2)) if x.dim() == 3 else x  # Matmul if 3D
        x = x **0.5
        return x


"""

    # Parse the code into an AST
    tree = ast.parse(code)

    # Transform the AST
    transformer = TorchMathTransformer()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # Convert the AST back to source code

    new_code = ast.unparse(new_tree)

    print(new_code)
