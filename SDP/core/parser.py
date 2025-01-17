from pyRDDLGym.XADD.RDDLModelXADD import RDDLModelWXADD

from SDP.core.action import Action
from SDP.core.mdp import MDP

import itertools


class Parser:
    def __init__(self):
        pass    

    def parse(
            self,
            model: RDDLModelWXADD,
            is_linear: bool = False,
            discount: float = 1.0
    ) -> MDP:
        """Parses the RDDL model into an MDP."""
        mdp = MDP(model, is_linear=is_linear, discount=discount)

        # Go through all actions and get corresponding CPFs and rewards
        actions = model.actions
        action_dict = {}
        action_type = {}
        for name, val in actions.items():
            atype = 'bool' if isinstance(val, bool) else 'real'
            ## only support 'bool' for now
            if atype == 'real':
                raise ValueError('continous action is not supported yet')
            a_symbol = model.ns.get(name)
            if a_symbol is None:
                print(f'Warning: action {name} not found in RDDLModelWXADD.actions')
                a_symbol, a_node_id = model.add_sympy_var(name, atype)
            action_dict[a_symbol] = False
            action_type[name] = atype
        
        action_names = list(actions.keys())
        action_symbols = [model.ns[n] for n in action_names]
        bool_combos = [list(i) for i in itertools.product([0, 1], repeat=len(action_names))]
        action_subs_list = []
        for b in bool_combos:
            a = {}
            for i, v in enumerate(b):
                a[action_symbols[i]] = True if v==1 else False
            action_subs_list.append(a)
        
        for a in action_subs_list:
            atype = 'bool'
            name = str(a)
            bool_dict = {}
            for k, v in a.items():
                bool_dict[str(k)] = v
            a_symbol = action_symbols
            action = Action(
                name, a_symbol, bool_dict, mdp.context, atype=atype, action_params=None
            )    # TODO: action_params (for continuous actions)
            subst_dict = a.copy()

            cpfs = model.cpfs

            for state_fluent, cpf in cpfs.items():
                cpf = action.restrict(cpf, subst_dict)
                var_ = model.ns[state_fluent]
                action.add_cpf(var_, cpf)


            reward = model.reward
            reward = action.restrict(reward, subst_dict)
            action.reward = reward

            mdp.add_action(action)
            
        return mdp
