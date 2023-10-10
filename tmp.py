import sympy as sp
from xaddpy import XADD

context = XADD()

# Get the unique ID of the decision expression
dec_expr: sp.Basic = sp.S('x + y <= 0')
dec_id, is_reversed = context.get_dec_expr_index(dec_expr, create=True)

# Get the IDs of the high and low branches: [0] and [2], respectively
high: int = context.get_leaf_node(sp.S(0))
low: int = context.get_leaf_node(sp.S(2))
if is_reversed:
    low, high = high, low

# Create the decision node with the IDs
dec_node_id: int = context.get_internal_node(dec_id, low=low, high=high)
# print(f"Node created:\n{context.get_repr(dec_node_id)}")

# dec_node_id = context.get_dec_node(dec_expr, low_val=sp.S(2), high_val=sp.S(0))

b = sp.Symbol('b', bool=True)
dec_b_id, _ = context.get_dec_expr_index(b, create=True)

high: int = context.get_leaf_node(sp.S(1))
node_id: int = context.get_internal_node(dec_b_id, low=dec_node_id, high=high)
print(f"Node created:\n{context.get_repr(node_id)}")

x, y = sp.symbols('x y')

bool_assign = {b: False}
cont_assign = {x: 2, y: -2}

res = context.evaluate(node_id, bool_assign=bool_assign, cont_assign=cont_assign)
print(f"Result: {res}")

subst_dict = {b: False}
node_id_after_subs = context.substitute(node_id, subst_dict)
print(f"Result:\n{context.get_repr(node_id_after_subs)}")

var_set = context.collect_vars(node_id)
print(f"var_set: {var_set}")
