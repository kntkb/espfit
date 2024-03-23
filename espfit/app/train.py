"""
Create espaloma network model and train the model.

TODO
----
* Add support to use multiple GPUs
* Improve how data are parsed using dataclasses or pydantic
"""
import os
import torch
import espaloma as esp
import logging

_logger = logging.getLogger(__name__)


class EspalomaBase(object):
    def __init__(self):
        # Check if GPU is available
        if torch.cuda.is_available():
            _logger.debug('GPU is available for training.')
        else:
            _logger.debug('GPU is not available for training.')

        # Check torch data type
        _logger.debug(f'Torch data type is {torch.get_default_dtype()}')


    @classmethod
    def from_toml(cls, filename, **override_espalomamodel_kwargs):
        """Create an instance of the class from a TOML configuration file.

        This method reads a TOML file specified by `filename`, extracts the 'espaloma'
        section of the configuration, and uses it to create a model. It then returns 
        an instance of the class initialized with this model. If the file is not found,
        the method prints the error and re-raises the exception.

        Parameters
        ----------
        filename : str
            Path to the TOML file containing the configuration for the espaloma model.

        override_espalomamodel_kwargs : dict
            A dictionary of keyword arguments to override the default settings for the 
            espaloma model.

        Returns
        -------
        object
            An instance of the class initialized with the model created from the 
            espaloma configuration in the TOML file.
        """
        import tomllib
        try:
            with open(filename, 'rb') as f:
                config = tomllib.load(f)
        except FileNotFoundError as e:
            print(e)
            raise

        model = cls()
        net = model.create_model(config['espaloma'])
        model.net = net
        model.configfile = filename

        # Update training settings
        for key, value in config['espaloma']['train'].items():
            if hasattr(model, key):
                setattr(model, key, value)
            else:
                raise ValueError(f'Invalid attribute {key}.')

        # Override training settings
        for key, value in override_espalomamodel_kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
            else:
                raise ValueError(f'Invalid attribute {key}.')

        return model


    @staticmethod
    def _get_base_module(espaloma_config):
        """Create base modules for Espaloma network model.
        
        Parameters
        ----------
        espaloma_config : dict
            A dictionary containing the configuration for the Espaloma network.
            This includes the method and options for the GNN layers, the configurations for the two stages of the network,
            and optionally the weights for different loss components.

        Returns
        -------
        list
            A list of modules for the Espaloma network model.
        """
        # GNN
        gnn_method = 'SAGEConv'
        gnn_options = {}
        for key in espaloma_config['gnn'].keys():
            if key == 'method':
                gnn_method = espaloma_config['gnn'][key]
            else:
                gnn_options[key] = espaloma_config['gnn'][key]
        layer = esp.nn.layers.dgl_legacy.gn(gnn_method, gnn_options)
        
        # Stage1
        config_1 = espaloma_config['nn']['stage1']
        representation = esp.nn.Sequential(layer, config=config_1)
        
        # Stage2
        config_2 = espaloma_config['nn']['stage2']
        units = config_2[0] 
        # out_features: Define modular MM parameters Espaloma will assign
        # 1: atom hardness and electronegativity
        # 2: bond linear combination, enforce positive
        # 3: angle linear combination, enforce positive
        # 4: torsion barrier heights (can be positive or negative)
        readout = esp.nn.readout.janossy.JanossyPooling(
            in_features=units, config=config_2,
            out_features={
                    1: {'s': 1, 'e': 1},
                    2: {'log_coefficients': 2},
                    3: {'log_coefficients': 2},
                    4: {'k': 6},
            },
        )
        # Improper torsions (multiplicity n=2)
        readout_improper = esp.nn.readout.janossy.JanossyPoolingWithSmirnoffImproper(in_features=units, config=config_2, out_features={"k": 2})

        # Initialize loss weights and update if provided
        weights = { 'energy': 1.0, 'force': 1.0, 'charge': 1.0, 'torsion': 1.0, 'improper': 1.0 }
        if 'weights' in espaloma_config.keys():
            for key in espaloma_config['weights'].keys():
                weights[key] = espaloma_config['weights'][key]

        # Append base modules
        modules = []
        modules.append(representation)
        modules.append(readout)
        modules.append(readout_improper)
        modules.append(esp.nn.readout.janossy.ExpCoefficients())
        modules.append(esp.nn.readout.charge_equilibrium.ChargeEquilibrium())

        return modules, weights


    @staticmethod
    def create_model(espaloma_config):
        """Create an Espaloma network model using the provided configuration.

        This function constructs a PyTorch Sequential model with two stages of Graph Neural Network (GNN) layers,
        JanossyPooling readout layers for various features, and additional layers for energy computation and loss calculation.
        The specifics of the GNN layers and the readout layers are controlled by the `espaloma_config` dictionary.
        If a CUDA-compatible GPU is available, the model is moved to the GPU before being returned.

        Parameters
        ----------
        espaloma_config : dict
            A dictionary containing the configuration for the Espaloma network.
            This includes the method and options for the GNN layers, the configurations for the two stages of the network,
            and optionally the weights for different loss components.

        Returns
        -------
        torch.nn.Sequential
            The constructed Espaloma network model.
        """
        from espfit.utils.espaloma.module import GetLoss

        # Get base model
        modules, weights = EspalomaBase._get_base_module(espaloma_config)

        # Define espaloma architecture
        modules.append(esp.mm.geometry.GeometryInGraph())
        modules.append(esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]))
        modules.append(GetLoss(weights))

        # Create model
        net = torch.nn.Sequential(*modules)
        if torch.cuda.is_available():
            return net.cuda()
        else:
            return net
        

    def save_model(self, net=None, checkpoint_file=None, output_model='espaloma.pt', output_directory_path=None):
        """Save the Espaloma network model to a file.
        
        This method saves the Espaloma network model to a file in the specified output directory.
        
        Parameters
        ----------
        net : torch.nn.Sequential
            The Espaloma network model to be saved. 
            `net` attribute of the class will be used if available.

        checkpoint_file : str
            The checkpoint exported during training that will be used to save the model.

        output_model : str, default='espaloma.pt'
            The output file name for the model.

        output_directory_path : str, default=None
            The directory where the model should be saved. 
            If not provided, the model will be saved in the current working directory.

        Returns
        -------
        None
        """
        if output_directory_path is not None:
            os.makedirs(output_directory_path, exist_ok=True)
        else:
            output_directory_path = os.getcwd()

        if self.net is not None:
            net = self.net

        if net:
            modules = []
            for module in net:
                if isinstance(module, esp.mm.geometry.GeometryInGraph):
                    break
                modules.append(module)
            modules.append(esp.nn.readout.janossy.LinearMixtureToOriginal())
            net = torch.nn.Sequential(*modules)
        else:
            raise ValueError('No model provided.')
        
        # Save model
        state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)
        torch.save(net, os.path.join(output_directory_path, output_model))


class EspalomaModel(EspalomaBase):
    """Espaloma network model and training modules.

    Methods
    -------
    from_toml(filename):
        Load espaloma configuration file in TOML format.
    
    Examples
    --------
    >>> from espfit.app.train import EspalomaModel
    >>> filename = 'espfit/data/config/config.toml'
    >>> # create espaloma network model from toml file
    >>> model = EspalomaModel.from_toml(filename)
    >>> # check espaloma network model
    >>> model.net
    >>> # load training dataset
    >>> model.dataset_train = ds
    >>> model.train()
    """

    def __init__(self, net=None, dataset_train=None, dataset_validation=None, dataset_test=None, 
                 epochs=1000, batch_size=128, learning_rate=1e-4, checkpoint_frequency=10, 
                 random_seed=2666, output_directory_path=None):
        """Initialize an instance of the class with an Espaloma network model and a random seed.

        This constructor method sets up the Espaloma network model, the training, validation, test datasets, 
        a configuratino file, and the random seed that will be used throughout the training process. 
        If no model or datasets are provided, the corresponding attributes will be set to None. If no random seed is 
        provided, the `random_seed` attribute will be set to 2666.

        Parameters
        ----------
        net : torch.nn.Sequential, default=None
            The Espaloma network model to be used for training.        
        
        dataset_train : espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The training dataset. espaloma.graphs.graph.Graph. If not provided, the `train_data` attribute will be set to None.

        dataset_validation : espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The validation dataset. If not provided, the `validation_data` attribute will be set to None.

        dataset_test : Dataset, espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The test dataset. If not provided, the `test_data` attribute will be set to None.

        epochs : int, default=1000
            The number of epochs to train the model for.

        batch_size : int, default=128
            The number of samples per batch.

        learning_rate : float, default=1e-4
            The learning rate for the optimizer.

        checkpoint_frequency : int, default=10
            The frequency at which the model should be saved.

        random_seed : int, default=2666
            The random seed used throughout the espaloma training.

        output_directory_path : str, default=None
            The directory where the model checkpoints should be saved. 
            If not provided, the checkpoints will be saved in the current working directory.
        """
        super(EspalomaBase, self).__init__()
        self.net = net
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.dataset_test = dataset_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_frequency = checkpoint_frequency
        self.restart_epoch = 0
        self.random_seed = random_seed
        if output_directory_path is None:
            output_directory_path = os.getcwd()
            self.output_directory_path = output_directory_path


    @property
    def output_directory_path(self):
        """Get output directory path."""
        return self._output_directory_path


    @output_directory_path.setter
    def output_directory_path(self, value):
        """Set output directory path."""
        self._output_directory_path = value
        # Create output directory if it does not exist
        os.makedirs(value, exist_ok=True)


    def report_loss(self, epoch, loss_dict):
        """Report loss.

        This method reports the loss at a given epoch to a log file.
        Each loss component is multiplied by 100 for better readability.

        Parameters
        ----------
        loss_dict : dict
            The loss trajectory that stores individual weighted losses at a given epoch.

        Returns
        -------
        None
        """
        import pandas as pd
        
        log_file_path = os.path.join(self.output_directory_path, 'reporter.log')
        df_new = pd.DataFrame.from_dict(loss_dict, orient='index').T
        df_new = df_new.mul(100)   # Multiple each loss component by 100. Is this large enough?
        df_new.insert(0, 'epoch', epoch)

        if os.path.exists(log_file_path):
            df_old = pd.read_csv(log_file_path, sep='\t')
            df = pd.concat([df_old, df_new])
        else:
            df = df_new
        df.to_csv(log_file_path, sep='\t', float_format='%.4f', index=False)


    def train(self):
        """
        Train the Espaloma network model.

        Returns
        -------
        None
        """
        from espfit.utils.units import HARTREE_TO_KCALPERMOL

        if self.dataset_train is None:
            raise ValueError('Training dataset is not provided.')
        
        # Load checkpoint
        self.restart_epoch = self._load_checkpoint()

        # Train
        # https://github.com/choderalab/espaloma/blob/main/espaloma/app/train.py#L33
        # https://github.com/choderalab/espaloma/blob/main/espaloma/data/dataset.py#L310            
        ds_tr_loader = self.dataset_train.view(collate_fn='graph', batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.restart_epoch, self.epochs):
                epoch = i + 1    # Start from epoch 1 (not zero-indexing)
                for g in ds_tr_loader:
                    optimizer.zero_grad()
                    # TODO: Better way to handle this?
                    if torch.cuda.is_available():
                        g = g.to("cuda:0")
                    g.nodes["n1"].data["xyz"].requires_grad = True 
                    #loss = self.net(g)
                    loss, loss_dict = self.net(g)
                    loss.backward()
                    optimizer.step()

                loss_dict['loss'] = loss.item()
                self.report_loss(epoch, loss_dict)

                if epoch % self.checkpoint_frequency == 0:
                    # Note: returned loss is a joint loss of different units.
                    loss = HARTREE_TO_KCALPERMOL * loss.pow(0.5).item()
                    _logger.info(f'Epoch {epoch}: loss={loss:.3f}')
                    self._save_checkpoint(epoch)
    
    
    def _train_sampler_old(self, sampler_patience=800, neff_threshold=0.2, sampler_weight=1.0, debug=False):
        """
        Train the Espaloma network model with sampler.

        Parameters
        ----------
        sampler_patience : int, default=800
            The number of epochs to wait before using sampler.

        neff_threshold : float, default=0.2
            The minimum effective sample size threshold.

        sampler_weight : float, default=1.0
            The weight for the sampler loss.

        debug : bool, default=False
            If True, use espaloma-0.3.pt for debugging.

        Returns
        -------
        None
        """
        from espfit.utils.sampler.reweight import SetupSamplerReweight

        # Note: RuntimeError will be raised if copy.deepcopy is used.
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace 
        # operation: [torch.cuda.FloatTensor [512, 1]], which is output 0 of AsStridedBackward0, is at version 2; 
        # expected version 1 instead. Hint: the backtrace further above shows the operation that failed to 
        # compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
        import copy
        net_copy = copy.deepcopy(self.net)

        self.sampler_patience = sampler_patience
        self.neff_threshold = neff_threshold

        if self.dataset_train is None:
            raise ValueError('Training dataset is not provided.')

        # Load checkpoint
        self.restart_epoch = self._load_checkpoint()

        # Initialize
        SamplerReweight = SetupSamplerReweight()
        
        # Train
        ds_tr_loader = self.dataset_train.view(collate_fn='graph', batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.restart_epoch, self.epochs):
                epoch = i + 1    # Start from 1 (not zero-indexing)
            
                # torch.cuda.OutOfMemoryError: CUDA out of memory. 
                # Tried to allocate 80.00 MiB (GPU 0; 10.75 GiB total capacity; 
                # 9.76 GiB already allocated; 7.62 MiB free; 10.40 GiB reserved in total by PyTorch) 
                # If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  
                # See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

                loss = torch.tensor(0.0)
                if torch.cuda.is_available():
                    loss = loss.cuda("cuda:0")

                for g in ds_tr_loader:
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        g = g.to("cuda:0")
                    g.nodes["n1"].data["xyz"].requires_grad = True

                    _loss, loss_dict = self.net(g)
                    loss += _loss

                if epoch > self.sampler_patience:
                    # Save checkpoint as local model (net.pt)
                    # `neff_min` is -1 if SamplerReweight.samplers is None
                    samplers = self._setup_local_samplers(epoch, net_copy, debug)
                    neff_min = SamplerReweight.get_effective_sample_size(temporary_samplers=samplers)

                    # If effective sample size is below threshold, update SamplerReweight.samplers and re-run simulaton
                    if neff_min < self.neff_threshold:
                        _logger.info(f'Minimum effective sample size ({neff_min:.3f}) below threshold ({self.neff_threshold})')
                        SamplerReweight.samplers = samplers
                        SamplerReweight.run()
                    del samplers

                    # Compute sampler loss
                    loss_list = SamplerReweight.compute_loss()   # list of torch.tensor
                    for sampler_index, sampler_loss in enumerate(loss_list):
                        sampler = SamplerReweight.samplers[sampler_index]
                        loss += sampler_loss * sampler_weight
                        loss_dict[f'{sampler.target_name}'] = sampler_loss.item()
                    loss_dict['neff'] = neff_min

                loss_dict['loss'] = loss.item()
                self.report_loss(epoch, loss_dict)

                # Back propagate
                loss.backward()
                optimizer.step()

                if epoch % self.checkpoint_frequency == 0:
                    # Note: returned loss is a joint loss of different units.
                    #_loss = HARTREE_TO_KCALPERMOL * loss.pow(0.5).item()
                    _logger.info(f'Epoch {epoch}: loss={loss.item():.3f}')
                    self._save_checkpoint(epoch)



    def train_sampler(self, sampler_patience=800, neff_threshold=0.2, sampler_weight=1.0, debug=False):
        """
        Train the Espaloma network model with sampler.

        Parameters
        ----------
        sampler_patience : int, default=800
            The number of epochs to wait before using sampler.

        neff_threshold : float, default=0.2
            The minimum effective sample size threshold.

        sampler_weight : float, default=1.0
            The weight for the sampler loss.

        debug : bool, default=False
            If True, use espaloma-0.3.pt for debugging.

        Returns
        -------
        None
        """
        from espfit.utils.sampler.reweight import SetupSamplerReweight

        # Note: RuntimeError will be raised if copy.deepcopy is used.
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace 
        # operation: [torch.cuda.FloatTensor [512, 1]], which is output 0 of AsStridedBackward0, is at version 2; 
        # expected version 1 instead. Hint: the backtrace further above shows the operation that failed to 
        # compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
        import copy
        net_copy = copy.deepcopy(self.net)

        self.sampler_patience = sampler_patience
        self.neff_threshold = neff_threshold

        if self.dataset_train is None:
            raise ValueError('Training dataset is not provided.')

        # Load checkpoint
        self.restart_epoch = self._load_checkpoint()

        # Initialize
        SamplerReweight = SetupSamplerReweight()
        
        # Train
        ds_tr_loader = self.dataset_train.view(collate_fn='graph', batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.restart_epoch, self.epochs):
                epoch = i + 1    # Start from 1 (not zero-indexing)
            
                #
                # Accumulate gradients to avoid CUDA out of memory error.
                # ---
                # torch.cuda.OutOfMemoryError: CUDA out of memory. 
                # Tried to allocate 80.00 MiB (GPU 0; 10.75 GiB total capacity; 
                # 9.76 GiB already allocated; 7.62 MiB free; 10.40 GiB reserved in total by PyTorch) 
                # If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  
                # See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

                # Check effective sample size and re-run simulation if necessary
                if epoch > self.sampler_patience:
                    # Save checkpoint as local model (net.pt)
                    # `neff_min` is -1 if SamplerReweight.samplers is None
                    samplers = self._setup_local_samplers(epoch, net_copy, debug)
                    neff_min = SamplerReweight.get_effective_sample_size(temporary_samplers=samplers)
                    if neff_min < self.neff_threshold:
                        _logger.info(f'Minimum effective sample size ({neff_min:.3f}) below threshold ({self.neff_threshold})')
                        SamplerReweight.samplers = samplers
                        SamplerReweight.run()
                    del samplers

                # Gradient accumulation
                accumulation_steps = len(ds_tr_loader)
                for g in ds_tr_loader:
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        g = g.to("cuda:0")
                    g.nodes["n1"].data["xyz"].requires_grad = True
                    # Compute loss
                    loss, loss_dict = self.net(g)
                    loss = loss/accumulation_steps
                    # MD sampler loss
                    if epoch > self.sampler_patience:
                        loss_list = SamplerReweight.compute_loss()   # list of torch.tensor
                        for sampler_index, sampler_loss in enumerate(loss_list):
                            sampler = SamplerReweight.samplers[sampler_index]
                            loss += (sampler_loss/accumulation_steps) * sampler_weight
                            loss_dict[f'{sampler.target_name}'] = sampler_loss.item()
                        loss_dict['neff'] = neff_min
                    loss.backward(retain_graph=True)
                loss_dict['loss'] = loss.item()
                self.report_loss(epoch, loss_dict)
                optimizer.step()
                
                if epoch % self.checkpoint_frequency == 0:
                    # Note: returned loss is a joint loss of different units.
                    #_loss = HARTREE_TO_KCALPERMOL * loss.pow(0.5).item()
                    _logger.info(f'Epoch {epoch}: loss={loss.item():.3f}')
                    self._save_checkpoint(epoch)


    def _load_checkpoint(self):
        """Load the last checkpoint and restart the training process.

        This method finds all the checkpoint files in the output directory, loads the 
        last checkpoint (e.g. net100.pt), and restarts the training process from the next step. 
        If no checkpoint files are found, the training process starts from the first step.

        Returns
        -------
        int
            The step from which the training process should be restarted.
        """
        import sys
        import glob

        checkpoints = glob.glob("{}/*.pt".format(self.output_directory_path))
        
        if checkpoints:
            n = [ int(c.split('ckpt')[1].split('.')[0]) for c in checkpoints ]
            n.sort()
            restart_epoch = n[-1]
            restart_checkpoint = os.path.join(self.output_directory_path, f"ckpt{restart_epoch}.pt")
            self.net.load_state_dict(torch.load(restart_checkpoint))
            logging.info(f'Restarting from ({restart_checkpoint}).')
        else:
            restart_epoch = 0
        
        if restart_epoch >= self.epochs:
            _logger.info(f'Already trained for {self.epochs} epochs.')
            sys.exit(0)
        elif restart_epoch > 0:
            _logger.info(f'Training for additional {self.epochs-restart_epoch} epochs.')
        else:
            _logger.info(f'Training from scratch for {self.epochs} epochs.')

        return restart_epoch


    def _save_checkpoint(self, epoch):
        """Save local model.

        Parameters
        ----------
        epoch : int
            The epoch number.

        Returns
        -------
        None
        """
        checkpoint_file = os.path.join(self.output_directory_path, f"ckpt{epoch}.pt")
        torch.save(self.net.state_dict(), checkpoint_file)


    def _save_local_model(self, epoch, net_copy):
        """Save local model (force field).
        
        Parameters
        ----------
        epoch : int
            The epoch number.

        net_copy : torch.nn.Sequential
            A deep copy of the Espaloma network model.

        Returns
        -------
        None
        """
        # Save checkpoint as temporary espaloma model (force field)
        _logger.info(f'Save ckpt{epoch}.pt as temporary espaloma model (net.pt)')
        self._save_checkpoint(epoch)
        local_model = os.path.join(self.output_directory_path, f"ckpt{epoch}.pt")
        self.save_model(net=net_copy, checkpoint_file=local_model, output_model=f"net.pt", output_directory_path=self.output_directory_path)


    def _setup_local_samplers(self, epoch, net_copy, debug):
        """Setup local samplers.
        
        Parameters
        ----------
        epoch : int
            The epoch number.

        net_copy : torch.nn.Sequential
            A deep copy of the Espaloma network model.
        
        debug : bool
            If True, use espaloma-0.3.2.pt for debugging.

        Returns
        -------
        list
            A list of sampler systems.
        """
        from espfit.app.sampler import SetupSampler
        
        self._save_local_model(epoch, net_copy)

        # Define sampler settings with override arguments
        args = [epoch]
        if debug == True:
            from importlib.resources import files
            small_molecule_forcefield = str(files('espfit').joinpath("data/forcefield/espaloma-0.3.2.pt"))
        else:
            small_molecule_forcefield = os.path.join(self.output_directory_path, f"net.pt")

        override_sampler_kwargs = { 
            "atomSubset": 'all',
            "small_molecule_forcefield": small_molecule_forcefield,
            "output_directory_path": self.output_directory_path
            }
        
        # Create sampler system from configuration file. Returns list of systems.                        
        samplers = SetupSampler._from_toml(self.configfile, *args, **override_sampler_kwargs)

        return samplers
    