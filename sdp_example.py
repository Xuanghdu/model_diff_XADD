
from pyRDDLGym.Core.Grounder.RDDLGrounder import RDDLGrounder
from pyRDDLGym.Core.Parser.parser import RDDLParser
from pyRDDLGym.Core.Parser.RDDLReader import RDDLReader
from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD
from xaddpy.xadd import XADD

from SDP.utils.utils import get_xadd_model_from_file
from SDP.core.parser import Parser
from SDP.core.mdp import MDP
from SDP.core.policy import Policy
from SDP.policy_evaluation.pe import PolicyEvaluation
from SDP.value_iteration.vi import ValueIteration

## Global Vars for SDP
DISCOUNT = 0.9
N_STEPS = 2

# Domain/Instance Path
f_domain = './RDDL/reservoir/reservoir_disc/domain.rddl'
f_instance = './RDDL/reservoir/reservoir_disc/instance_1res_source.rddl'

# load xadd model and context see SDP.utils for details
model, context = get_xadd_model_from_file(f_domain, f_instance)


### Value Iteration
parser = Parser()
mdp = parser.parse(model, is_linear=True, discount=DISCOUNT) ## SDP currently only handle linear cases

vi= ValueIteration(mdp, N_STEPS)
value_id_vi, q_id_list_vi = vi.solve()

print(vi.print(value_id_vi))

### Policy PolicyEvaluation
parser = Parser()
mdp = parser.parse(model, is_linear=True, discount=DISCOUNT) ## SDP currently only handle linear cases

# need to definte a policy by a string or load from xadd file
policy_str_release_true =  "( [rlevel___t1 - 55 <= 0] ( [0] ) ( [1] ) )"
policy_str_release_false =  "( [rlevel___t1 - 55 <= 0] ( [1] ) ( [0] ) )"
# ## import using: 
# policy_str_release_true = context.import_xadd('release___t1_true.xadd', locals=context._str_var_to_var)

# get node ids for xadd
policy_id_release_true = context.import_xadd(xadd_str=policy_str_release_true)
policy_id_release_false = context.import_xadd(xadd_str=policy_str_release_false)

# make a dictionary of action as stringst to node id
xadd_policy = {
    '{release___t1: True}': policy_id_release_true,
    '{release___t1: False}': policy_id_release_false,
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







