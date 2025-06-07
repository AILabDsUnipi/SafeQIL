import copy

from SCOPIL.experiments.train_icrl_experiment import ICRLExperiment


class VICRLExperiment(ICRLExperiment):
    def __init__(
            self,
            environment,
            agent=None,
            config=None,
            file_results_dir="./tmp",
            seed=None
    ):
        # Create an 'ICRL' field and copy the 'VICRL' since the superclass (ICRLExperiment)
        # gets the hyperparameters from 'ICRL' field
        config['ICRL'] = copy.deepcopy(config['VICRL'])

        super().__init__(environment, agent, config, file_results_dir, seed)