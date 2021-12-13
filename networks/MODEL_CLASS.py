import tensorflow as tf
import os


class MODEL():
    def __init__(self,
                output_dir, #dir to save model/logs/checkpoints to
                name='network', #name of model for logging/saving purposes
                model_args={}, #arguments to pass into model function
                optimizer_args={} #arguments to pass into optimizerf function
                ):
        self.output_dir = output_dir
        self.name = name
        self.model = self._build_model(**model_args)
        self.optimizer = self._build_optimizer(**optimizer_args)
        self.logger = tf.summary.create_file_writer(os.path.join(self.output_dir, 'logs/train/', f'{self.name}'))
        self.val_logger = tf.summary.create_file_writer(os.path.join(self.output_dir, 'logs/val/', f'{self.name}'))
        self.test_logger = tf.summary.create_file_writer(os.path.join(self.output_dir, 'logs/test/', f'{self.name}'))

    #call method, takes in input and runs it through model call
    def __call__(self, input, **kwargs):
        return self.model(input, **kwargs)

    #method to save the model
    def save_model(self,
                save_path=None, #path to save to, if None, saved to default path
                **kwargs #kwargs for save function, if compile is not included, compile is set to false
                ):
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'saved_models', f'{self.name}/')
        self.model.save(save_path, **kwargs)

    #method to load the model
    def load_model(self,
                load_path=None, #path to load from, if None, loaded from default path
                name=None, #name to load from, if None, default name is taken
                **kwargs #kwargs for loading function, if compile is not included, set false
                ):
        if name is None:
            name = self.name
        if 'compile' not in kwargs:
            kwargs.update(compile = False)
        if load_path is None:
            load_dir = os.path.join(self.output_dir, 'saved_models', f'{name}/')
        else:
            load_dir = os.path.join(load_path, 'saved_models', f'{name}/')
        if os.path.exists(load_dir):
            self.model = tf.keras.models.load_model(load_dir **kwargs)
        else:
            raise OSError("Model Folder Does Not Exist!")

    #method to get the summary of the model
    def summary(self):
        self.model.summary()

    #method to build the model, should be overwritten by Subclass
    def _build_model(self, **kwargs):
        return tf.keras.Model(name=self.name)

    #method to build the model optimizer, should be overwritten by Subclass
    def _build_optimizer(self, **kwargs):
        return tf.keras.optimizers.Optimizer(f'{self.name}_optimizer')
