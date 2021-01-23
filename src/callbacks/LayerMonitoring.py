from stable_baselines3.common.callbacks import BaseCallback


class LayerActivationMonitoring(BaseCallback):
    def _on_step(self) -> bool:
        pass

    def __init__(self, verbose=0):
        super(LayerActivationMonitoring, self).__init__(verbose)

    def _on_rollout_start(self) -> None:
        self.model.policy.features_extractor.cnn.eval()
        self.model.policy.features_extractor.linear.eval()

        # Remove the hooks so that they don't get called for rollout collection
        hooks = self.model.policy.features_extractor.hooks
        hooks.remove()

        for i, hook in enumerate(hooks):
            if hook.activation_data is not None:
                data = hook.activation_data

                self.logger.record(f'diagnostics/activation_l{i}', data)
                self.logger.record(f'diagnostics/mean_l{i}', data.mean().item())
                self.logger.record(f'diagnostics/std_l{i}', data.std().item())
                self.logger.record(f'diagnostics/low_act_prop_l{i}', self.get_low_act(data).item())

    def _on_rollout_end(self) -> None:
        self.model.policy.features_extractor.cnn.train()
        self.model.policy.features_extractor.linear.train()

        # Add hooks to monitor model
        self.model.policy.features_extractor.hooks.register()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def get_low_act(self, data, threshold=0.2):
        low_activation = ((-threshold <= data) & (data <= threshold))
        return low_activation.count_nonzero() / low_activation.numel()
