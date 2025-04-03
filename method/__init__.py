class MetaAgent:
    def act(self, *args, **kwargs):
        raise NotImplementedError

    def process_observation(self, *args, **kwargs):
        raise NotImplementedError

    def post_process_action(self, *args, **kwargs):
        return

    def reset(self):
        return