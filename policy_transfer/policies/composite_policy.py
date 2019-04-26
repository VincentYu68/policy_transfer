


class CompositePolicy(object):
    recurrent = False
    def __init__(self, policies, mix_ratio, input_dims):
        self.policies = policies
        self.mix_ratio = mix_ratio
        self.input_dims = input_dims

        assert(len(mix_ratio) == len(policies))

    def act(self, stochastic, ob, is_training=False):
        ac, vpred = self.policies[0].act(stochastic, ob[0:self.input_dims[0]])
        ac = ac * self.mix_ratio[0]
        vpred = vpred * self.mix_ratio[0]

        for i in range(1, len(self.policies)):
            aci, v = self.policies[i].act(stochastic, ob[0:self.input_dims[i]])
            ac += aci * self.mix_ratio[i]
            vpred += v * self.mix_ratio[i]
        return ac, vpred

    def get_variables(self):
        pass
    def get_trainable_variables(self):
        pass
    def get_initial_state(self):
        return []

