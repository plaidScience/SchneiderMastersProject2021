import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='SG_LR_Scheduler', name=None)
class StarGANSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, start_step, decay_steps):
        super(StarGANSchedule, self).__init__()

        self.initial = initial_lr
        self.start_step = start_step

        self.decay_steps = decay_steps

    def __call__(self, step):
        computed = self.initial-((step+1)-self.start_step)*(self.initial/self.decay_steps)
        return tf.clip_by_value(computed, 0.0, self.initial)
    def get_config(self):
        return {
            'initial_lr':self.initial,
            'start_step':self.start_step,
            'decay_steps':self.decay_steps
        }