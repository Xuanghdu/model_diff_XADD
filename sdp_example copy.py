
from SDP.core.mdp import MDP
from SDP.core.parser import Parser
from SDP.core.policy import Policy
from SDP.policy_evaluation.pe import PolicyEvaluation
from SDP.utils.utils import get_xadd_model_from_file
from SDP.value_iteration.vi import ValueIteration

## Global Vars for SDP
# DISCOUNT = 0.9
DISCOUNT = 1
N_STEPS = 4

# Domain/Instance Path
# f_domain = './RDDL/reservoir/reservoir_disc/domain.rddl'
# f_instance = './RDDL/reservoir/reservoir_disc/instance_1res_source.rddl'
f_domain = './RDDL/robot/domain_new.rddl'
f_instance = './RDDL/robot/instance.rddl'

# load xadd model and context see SDP.utils for details
model, context = get_xadd_model_from_file(f_domain, f_instance)

# ### Value Iteration
# parser = Parser()
# mdp = parser.parse(model, is_linear=True, discount=DISCOUNT) ## SDP currently only handle linear cases

# vi= ValueIteration(mdp, N_STEPS)
# value_id_vi, q_id_list_vi = vi.solve()

# # Visualize XADD
# context.save_graph(value_id_vi, f"./robot_vi_{N_STEPS}_{DISCOUNT}.pdf")

# # can printout value XADD using print function in VI
# print(vi.print(value_id_vi))

# # print(q_id_list_vi)

### Policy PolicyEvaluation
parser = Parser()
mdp = parser.parse(model, is_linear=True, discount=DISCOUNT) ## SDP currently only handle linear cases

for aname, action in mdp.actions.items():
    print(aname, action)

# # need to definte a policy by a string or load from xadd file
policy_str_right_true = "( [pos_x_robot - 5 <= 0] ( [1] ) ( [0] ) )"
policy_str_up_true = "( [pos_x_robot - 5 <= 0] ( [0] ) ( [1] ) )"
policy_str_right_true_up_true   =  "( [0] )"
policy_str_right_false_up_false  =  "( [0] )"

# ### import using:
# policy_str_release_true = context.import_xadd('release___t1_true.xadd', locals=context._str_var_to_var)

# get node ids for xadd
policy_str_right_true = context.import_xadd(xadd_str=policy_str_right_true)
policy_str_up_true = context.import_xadd(xadd_str=policy_str_up_true)
policy_str_right_true_up_true = context.import_xadd(xadd_str=policy_str_right_true_up_true)
policy_str_right_false_up_false = context.import_xadd(xadd_str=policy_str_right_false_up_false)

# make a dictionary of action as stringst to node id
xadd_policy = {
    '{up: False, right: False}': policy_str_right_false_up_false,
    '{up: False, right: True}': policy_str_right_true,
    '{up: True, right: False}': policy_str_up_true,
    '{up: True, right: True}': policy_str_right_true_up_true,
}


# load policy to mdp class
policy = Policy(mdp)
policy_dict = {}
for aname, action in mdp.actions.items():
    policy_dict[action] = xadd_policy[aname]
policy.load_policy(policy_dict)

## do policy evluation for n steps
pe = PolicyEvaluation(mdp, policy,N_STEPS)
value_id_pe, q_id_list_pe = pe.solve()

# can printout value XADD using print function in pe
print(pe.print(value_id_pe))
# print(q_id_list_pe)
# print(q_id_list_pe[0])
# print(context._id_to_node[q_id_list_pe[0]])

#
# print(f"XADD: \n{context.get_repr()}")

# Visualize XADD
context.save_graph(value_id_pe, f"./robot_pe_area_tmp_{N_STEPS}_{DISCOUNT}.pdf")

# Visualize value function
# plot a grid of value function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = np.arange(0, 10, 1)
y = np.arange(0, 10, 1)

X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

var_set = context.collect_vars(value_id_pe)
var_dict = {}

# pos_y_robot, pos_x_robot, reach_counter = var_set
for i in var_set:
    var_dict[f"{i}"] = i

print(var_dict)

for i in range(len(x)):
    for j in range(len(y)):
        cont_assign = {var_dict["pos_x_robot"]: x[i], var_dict["pos_y_robot"]: y[j]}
        bool_assign = {var_dict["reach_counter"]: False}
        Z[i][j] = context.evaluate(value_id_pe, bool_assign=bool_assign, cont_assign=cont_assign)

# plt.close('all')
# plt.clf()

# matplotlib.use('GTK4Agg', force=True)

fig, ax = plt.subplots()

im = ax.imshow(Z, cmap='hot', interpolation='nearest')
ax.set_title(f'Value Function for {N_STEPS} steps')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(im)
ax.set_xticks(np.arange(len(x)))
ax.set_yticks(np.arange(len(y)))
plt.show()
plt.savefig(f"./robot_value_{N_STEPS}_{DISCOUNT}.png")
