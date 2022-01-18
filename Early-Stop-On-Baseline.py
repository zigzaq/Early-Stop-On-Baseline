from tensorflow.keras.callbacks import Callback


class EarlyStopOnBaseline(Callback):
    """Callback that terminates training when either acc, loss, val_acc or val_loss reach a specified baseline
    (str) monitor - what to monitor: available for now: acc, loss, val_acc, val_loss; default: val_loss
    (float) baseline - expected value to reach before monitoring progress by delta; default: 0.1
    (float) delta - how much the monitored value has to decrease/increase (loss/acc); default: 0.01
    (int) patience - how many epochs with no better results to wait before stopping; default: 0
    """
    def __init__(self, monitor: str = 'val_loss', baseline: float = 0.1, min_delta: float = 0.01, patience: int = 0):
        super(EarlyStopOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.delta = min_delta
        self.patience = patience
        self._patience = 0
        self._flag_monitor_accuracy = any([a in monitor for a in ['acc', 'accuracy']])
        self._best_value = 1e10
        # assume loss is the monitored value
        if self._flag_monitor_accuracy:
            self._best_value = 1e-10
        self._flag_reached_baseline = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        monitored_value = logs.get(self.monitor)

        if monitored_value is None or self.baseline is None:
            return

        self._flag_reached_baseline = monitored_value <= self.baseline or self._flag_reached_baseline
        flag_better_result = monitored_value <= self.best_value - self.delta
        if self._flag_monitor_accuracy:
            self._flag_reached_baseline = monitored_value >= self.baseline or self._flag_reached_baseline
            flag_better_result = monitored_value >= self.best_value + self.delta

        if not self.flag_reached_baseline:
            return

        if flag_better_result:
            self._best_value = monitored_value
            self._patience = 0
            return

        self._patience += 1
        if self._patience < self.patience:
            return

        print(f'\nEpoch {epoch}: Terminating training.')
        self.model.stop_training = True


if __name__ == '__main__':
    stop = EarlyStopOnBaseline()
    print(stop)
