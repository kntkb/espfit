"""
Create MD system and perform MD simulation.

Notes
-----
* Use Interchange to create biopolymer-ligand system when it supports RNA systems. Currently (2024-01-09), OpenFF Toolkit identifies residues 
by matching chemical substructures rather than by residue name, it currently only supports the 20 'canonical' amino acids.
* Template keywords for EspalomaTemplateGenerator are hard coded to avoid unexpected change in default settings of EspalomaTemplateGenerator.
template_generator_kwargs = {"reference_forcefield": "openff_unconstrained-2.0.0", "charge_method": "nn"}
* The `self._system_generator` in #L437 is reconstructed to regenerate the system with espaloma to avoid system construction errors for some 
stored testsystems (espfit/data/target/testsystems/nucleoside).

TODO
----
* Improve system construction with espaloma. Currently, the system construction with espaloma is not stable for some test systems and 
requires work arounds.
* Improve the way to handle multiple biopolymer chains for constructing systems with espaloma.
"""
import os
import openmm.app as app
import openmm.unit as unit
import logging

_logger = logging.getLogger(__name__)


class BaseSimulation(object):
    """Base class for MD sampler.

    Methods
    -------
    minimize(output_directory_path=None):
        Minimize solvated system.
    
    run(output_directory_path=None):
        Run standard MD simulation.

    export_xml(exportSystem=True, exportState=True, exportIntegrator=True, output_directory_path=None):
        Export serialized system XML file and solvated pdb file.
    """
    def __init__(self, maxIterations=100, nsteps=2500000, atomSubset='solute', 
                 checkpoint_frequency=250000, logging_frequency=250000, netcdf_frequency=250000, 
                 output_directory_path=None, input_directory_path=None):
        """Initialize base simulation object.
        
        Parameters
        ----------
        maxIterations : int, default=100
            Maximum number of iterations to perform minimization.

        nsteps : int, default=2500000 (10 ns using 4 fs timestep)
            Number of steps to run the simulation.

        atomSubset : str, default='solute'
            Subset of atoms to save. Default is 'solute'. Other options 'all' and 'not water'.
            
        checkpoint_frequency : int, default=250000 (1 ns)
            Frequency (in steps) at which to write checkpoint files.

        logging_frequency : int, default=250000 (1 ns)
            Frequency (in steps) at which to write logging files.

        netcdf_frequency : int, default=250000 (1 ns)
            Frequency (in steps) at which to write netcdf files.

        output_directory_path : str, optional
            Output directory path. Default is None.
            If None, the current working directory will be used.

        input_directory_path : str, optional
            Input directory path to restart simulation. Default is None.
            If None, the current working directory will be used.
        """
        self.maxIterations = maxIterations
        self.nsteps = nsteps
        self.atomSubset = atomSubset
        self.checkpoint_frequency = checkpoint_frequency
        self.logging_frequency = logging_frequency
        self.netcdf_frequency = netcdf_frequency

        if self.atomSubset not in ['solute', 'all', 'not water']:
            raise ValueError(f"Invalid atomSubset: {self.atomSubset}. Expected 'solute', 'all', or 'not water'.")

        if output_directory_path is None:
            output_directory_path = os.getcwd()  # Is this right?
        if input_directory_path is None:
            input_directory_path = os.getcwd()
        
        self.output_directory_path = output_directory_path   # TODO: Is the property decorator and setter properly defined
        self.input_directory_path = input_directory_path
        self.platform = self._get_platform()


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


    def _get_platform(self):
        """Get fastest platform. 
        
        Notes
        -----
        Set CUDA DeterministicForces to true for accurate reweighting using MBAR.
        See the following for more details:
        * http://docs.openmm.org/latest/userguide/library/04_platform_specifics.html
        * https://github.com/openmm/openmm/issues/1947#issuecomment-350119490
                
        Returns
        -------
        platform : object
            OpenMM platform
        """
        from openmmtools.utils import get_fastest_platform
        platform = get_fastest_platform()
        platform_name = platform.getName()
        _logger.debug(f"Fastest platform: {platform_name}")
        if platform_name == "CUDA":
            platform.setPropertyDefaultValue('DeterministicForces', 'true')  # default is false
            platform.setPropertyDefaultValue('Precision', 'mixed')  # default is single
        
        return platform
    

    def minimize(self, output_directory_path=None):
        """Minimize solvated system.

        output_directory_path : str, default=None
            The path to the output directory. If None, the default output directory is used.

        Returns
        -------
        None
        """
        if output_directory_path is not None:
            self.output_directory_path = output_directory_path  # property decorator is called

        _logger.debug(f"Minimizing system for maximum {self.maxIterations} steps")
        self.simulation.minimizeEnergy(self.maxIterations)


    def run(self, output_directory_path=None):
        """Run standard MD simulation.

        Parameters
        ----------
        output_directory_path : str, default=None
            The path to the output directory. If None, the default output directory is used.

        Returns
        -------
        None
        """
        import mdtraj
        from mdtraj.reporters import NetCDFReporter
        from openmm.app import CheckpointReporter, StateDataReporter
        
        if output_directory_path is not None:
            self.output_directory_path = output_directory_path  # property decorator is called

        # Select atoms to save
        if self.atomSubset == 'all':
            self.atom_indices = None
        else:
            self.atom_indices = []
            mdtop = mdtraj.Topology.from_openmm(self.simulation.topology)
            if self.atomSubset == 'solute':
                res = [ r for r in mdtop.residues if r.name not in ('HOH', 'NA', 'CL', 'K') ]
            elif self.atomSubset == 'not water':
               res = [ r for r in mdtop.residues if r.name not in ('HOH') ]
            for r in res:
                for a in r.atoms:
                    self.atom_indices.append(a.index)
       
        # Define reporter
        self._check_file_exists("traj.nc")
        self.simulation.reporters.append(NetCDFReporter(os.path.join(self.output_directory_path, f"traj.nc"), 
                                                        min(self.netcdf_frequency, self.nsteps), 
                                                        atomSubset=self.atom_indices))
        self._check_file_exists("checkpoint.chk")
        self.simulation.reporters.append(CheckpointReporter(os.path.join(self.output_directory_path, f"checkpoint.chk"), 
                                                            min(self.checkpoint_frequency, self.nsteps)))
        self._check_file_exists("reporter.log")
        self.simulation.reporters.append(StateDataReporter(os.path.join(self.output_directory_path, f"reporter.log"), 
                                                           min(self.logging_frequency, self.nsteps), 
                                                           step=True, potentialEnergy=True, kineticEnergy=True, 
                                                           totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
        
        # Run
        _logger.info(f"Running simulation for {self.nsteps} steps...")
        self.simulation.step(self.nsteps)


    def export_xml(self, exportSystem=True, exportState=True, exportIntegrator=True, output_directory_path=None):
        """Export serialized system XML file and solvated pdb file.

        TODO
        ----
        * Currently, the output filenames are hard-coded. Should output filenames be specified by users?

        Parameters
        ----------
        exportSystem : bool, default=True
            Whether to export system XML file.

        exportState : bool, default=True
            Whether to export state XML file.

        exportIntegrator : bool, default=True
            Whether to export integrator XML file.

        output_directory_path : str, default=None
            The path to the output directory. If None, the default output directory is used.

        Returns
        -------
        None
        """
        from openmm import XmlSerializer
        _logger.debug(f"Serialize and export system")

        if output_directory_path is not None:
            # Create a new output directory different from the one specified when the SetupSampler instance was created.
            self.output_directory_path = output_directory_path   # property decorator is called

        state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)

        # Save system
        if exportSystem:
            self._check_file_exists("system.xml")
            outfile = os.path.join(self.output_directory_path, f"system.xml")
            with open(f"{outfile}", "w") as wf:
                xml = XmlSerializer.serialize(self.simulation.system)
                wf.write(xml)
        
        # Save and serialize the final state
        if exportState:
            self._check_file_exists("state.xml")            
            outfile = os.path.join(self.output_directory_path, f"state.xml")
            with open(f"{outfile}", "w") as wf:
                xml = XmlSerializer.serialize(state)
                wf.write(xml)

            # Save as pdb file
            self._check_file_exists("state.pdb")
            outfile = os.path.join(self.output_directory_path, f"state.pdb")
            with open(f"{outfile}", "w") as wf:
                app.PDBFile.writeFile(
                    self.simulation.topology,
                    self.simulation.context.getState(
                        getPositions=True,
                        #enforcePeriodicBox=False).getPositions(),
                        enforcePeriodicBox=True).getPositions(),
                        file=wf,
                        keepIds=True
                )
        
        # Save and serialize integrator
        if exportIntegrator:
            self._check_file_exists("integrator.xml")
            outfile = os.path.join(self.output_directory_path, f"integrator.xml")
            with open(f"{outfile}", "w") as wf:
                xml = XmlSerializer.serialize(self.simulation.integrator)
                wf.write(xml)


    def _check_file_exists(self, filename):
        """Renumber the given filename if it already exists in the output directory.

        Parameters
        ----------
        filename : str
            The name of the file to be renumbered.

        Returns
        -------
        None
        """
        import glob
        import shutil

        if os.path.exists(os.path.join(self.output_directory_path, filename)):
            basename, extension = os.path.splitext(filename)
            index = len(glob.glob(os.path.join(self.output_directory_path, f"{basename}*{extension}")))
            
            _logger.info(f"File {filename} already exists. Rename to {basename}{index}{extension}.")
            shutil.copy(
                os.path.join(self.output_directory_path, filename),
                os.path.join(self.output_directory_path, f"{basename}{index}{extension}")
            )


class SetupSampler(BaseSimulation):
    """Create biopolymer-ligand system.

    Use espaloma force field as default to self-consistently parameterize the biopolymer-ligand system.
    Use Perses 0.10.1 default parameter settings to setup the system.

    Methods
    -------
    create_system(biopolymer_file=None, ligand_file=None):
        Create biopolymer-ligand system and export serialized system XML file and solvated pdb file.

    from_toml(filename, *args, **override_sampler_kwargs):
        Create SetupSampler from a TOML configuration file.

    from_xml(filename):
        Create SetupSampler from a serialized system XML file.

    Examples
    --------
    >>> from espfit.app.sampler import SetupSampler
    >>> c = SetupSampler()
    >>> c.create_system(biopolymer_file='protein.pdb', ligand_file='ligand.sdf')
    >>> c.maxIterations = 10   # change default setting
    >>> c.minimize()
    >>> c.nsteps = 100         # change default setting
    >>> c.run()

    Notes
    -----
    For some reason, the following force field files fail to construct systems for test systems stored in `espfit/data/target/testsystems`:

    ['amber14-all.xml', 'amber/phosaa14SB.xml']             : pl-multi (TPO): NG, pl-single: NG, RNA: NG
    ['amber/protein.ff14SB.xml', 'amber/phosaa14SB.xml']    : pl-multi (TPO): NG, pl-single: NG, RNA: NG
    ['amber14-all.xml']                                     : pl-multi (TPO): NG, pl-single: OK, RNA: OK
    ['amber/protein.ff14SB.xml', 'amber/RNA.OL3.xml']       : pl-multi (TPO): NG, pl-single: OK, RNA: OK
    """
    def __init__(self, 
                 small_molecule_forcefield='espaloma-0.3.2',
                 forcefield_files = ['amber/ff14SB.xml', 'amber/phosaa14SB.xml'],
                 water_model='tip3p', 
                 solvent_padding=9.0 * unit.angstroms, 
                 ionic_strength=0.15 * unit.molar, 
                 hmass=3.0 * unit.amu, 
                 temperature=300.0 * unit.kelvin, 
                 pressure=1.0 * unit.atmosphere, 
                 pme_tol=2.5e-04, 
                 barostat_period=50, 
                 timestep=4 * unit.femtoseconds, 
                 override_with_espaloma=True,
                 **kwargs
                 ):
        """Initialize SetupSampler.
        
        Parameters
        ----------
        small_molecule_forcefield : str, optional
            The force field to be used for small molecules. Default is 'espaloma-0.3.2'.
            Alternative recommended choice is 'openff-2.1.0'.
        forcefield_files : list, optional
            List of force field files. Default is ['amber14-all.xml'].
        water_model : str, optional
            The water model to be used. Default is 'tip3p'.
        solvent_padding : Quantity, optional
            The padding distance around the solute in the solvent box. Default is 9.0 * unit.angstroms.
        ionic_strength : Quantity, optional
            The ionic strength of the solvent. Default is 0.15 * unit.molar.
        hmass : Quantity, optional
            The mass of the hydrogen atoms. Default is 3.0 * unit.amu.
        temperature : Quantity, optional
            The temperature of the system. Default is 300.0 * unit.kelvin.
        pressure : Quantity, optional
            The pressure of the system. Default is 1.0 * unit.atmosphere.
        pme_tol : float, optional
            The Ewald error tolerance for PME electrostatics. Default is 2.5e-04.
        barostat_period : int, optional
            The frequency at which the barostat is applied. Default is 50.
        timestep : Quantity, optional
            The integration timestep. Default is 4 * unit.femtoseconds.
        override_with_espaloma : bool, optional
            Whether to override the original parameters with espaloma. Default is True.
            This will override all solute molecules with espaloma parameters.
        """
        super(SetupSampler, self).__init__(**kwargs)
        if small_molecule_forcefield == 'espaloma-0.3.2':
            from importlib.resources import files
            small_molecule_forcefield = str(files('espfit').joinpath("data/forcefield/espaloma-0.3.2.pt"))
        self.small_molecule_forcefield = small_molecule_forcefield
        self.water_model = water_model
        self.forcefield_files = forcefield_files
        self.solvent_padding = solvent_padding
        self.ionic_strength = ionic_strength
        self.hmass = hmass
        self.temperature = temperature
        self.pressure = pressure
        self.pme_tol = pme_tol
        self.barostat_period = barostat_period
        self.timestep = timestep
        self.override_with_espaloma = override_with_espaloma
        self.target_class = None
        self.target_name = None
        # Update forcefield file list to add water model files
        self._update_forcefield_files()
        # Get water class (3-site: tip3p, 4-site model: tip4pew)
        self._get_water_class()


    @classmethod
    def _from_toml(cls, filename, *args, **override_sampler_kwargs):
        """Create SetupSampler from a TOML configuration file.
        
        Note that this is designed for creating new systems with temporary espaloma models generated during 
        espaloma training. It supports multiple systems in the configuration file and returns a list of 
        SetupSampler instances.

        Parameters
        ----------
        filename : str
            The path to the TOML configuration file.

        *args : list
            This is used to update the output directory path during espaloma training.
            The list should contain a single integer value, corresponding to the epoch number.
            
        **override_sampler_kwargs : dict
            The dictionary of keyword arguments to override the default settings of the 
            BaseSimulation and SetupSampler classes. This option is intended for creating
            new systems with temporary espaloma models generated during espaloma training.

        Returns
        -------
        samplers : list of SetupSampler instances
        """
        import tomllib
        from espfit.utils.units import convert_string_to_unit
        from importlib.resources import files

        try:
            with open(filename, 'rb') as f:
                config = tomllib.load(f)
        except FileNotFoundError as e:
            print(e)
            raise
        
        config = config['sampler']['setup']  # list of multiple setups
        if config is None:
            raise ValueError("target is not specified in the configuration file")
        
        # If the configuration file contains a single system, convert it to a list
        if not isinstance(config, list):
            config = [config]

        samplers = []
        _logger.debug(f'Found {len(config)} systems in the configuration file')
        for _config in config:
            # Create SetupSampler instance with default settings
            # Note that this automatically adds the default water model (tip3p) to `self.forcefield_files`.
            sampler = cls()

            # Target information
            target_class = _config.get('target_class', None)
            target_name = _config.get('target_name', None)
            sampler.target_class = target_class
            sampler.target_name = target_name

            # Get biopolymer and ligand file if given
            # Priority 1: Use input files if given
            biopolymer_file, ligand_file = None, None
            if _config.get('biopolymer_file'):
                biopolymer_file = _config['biopolymer_file']
                if not os.path.exists(biopolymer_file):
                    raise FileNotFoundError(f"File not found: {biopolymer_file}")
            if _config.get('ligand_file'):
                ligand_file = _config['ligand_file']
                if not os.path.exists(ligand_file):
                    raise FileNotFoundError(f"File not found: {ligand_file}")
            # Priority 2: Search espfit/data/target directory if input files are not given
            if biopolymer_file is None and ligand_file is None:
                biopolymer_file = files('espfit').joinpath(f'data/target/{target_class}/{target_name}/target.pdb')
                ligand_file = files('espfit').joinpath(f'data/target/{target_class}/{target_name}/ligand.sdf')
                if not biopolymer_file.exists():
                    raise FileNotFoundError(f"File not found: {biopolymer_file}")
                if not ligand_file.exists():
                    ligand_file = None

            # System settings
            for key, value in _config.items():
                if key not in ['target_class', 'target_name', 'biopolymer_file', 'ligand_file']:
                    if hasattr(sampler, key):
                        if isinstance(value, str) and "*" in value:
                            _value = float(value.split('*')[0].strip())
                            unit_string = value.split('*')[1].strip()
                            unit_mapping = convert_string_to_unit(unit_string)
                            value = _value * unit_mapping                        
                        setattr(sampler, key, value)
                    else:
                        raise ValueError(f"Invalid keyword argument: {key}")

            # Pass temporary espaloma model to the sampler if kwargs are given
            for key, value in override_sampler_kwargs.items():
                if hasattr(sampler, key):
                    setattr(sampler, key, value)
                else:
                    raise ValueError(f"Invalid keyword argument: {key}")

            # Update output directory path if args (epoch) is given
            if args:
                if len(args) == 1 and isinstance(args[0], int):
                    sampler.output_directory_path = os.path.join(sampler.output_directory_path, sampler.target_name, f'{args[0]}')
                else:
                    raise ValueError(f"Invalid argument: {args}. Expected a single integer value for the epoch number.")

            # Empty and recreate `forcefield_files` to avoid appending multiple water model forcefields.
            # Use list of forcefield files if given in the `override_sampler_kwargs`. If not given, use the default forcefield_files.
            if 'forcefield_files' in override_sampler_kwargs.keys():
                pass
            else:
                forcefield_files = ['amber/ff14SB.xml', 'amber/phosaa14SB.xml']
            sampler.forcefield_files = forcefield_files
            sampler._update_forcefield_files()
            # Update water class (3-site: tip3p, 4-site model: tip4pew)
            sampler._get_water_class()

            # Create system
            sampler.create_system(biopolymer_file=biopolymer_file, ligand_file=ligand_file)            
            samplers.append(sampler)
            del sampler
        
        return samplers


    def _update_forcefield_files(self):
        """Get forcefield files.

        Update `forcefield_files` depending on the type of water model.

        Returns
        -------
        updated_forcefield_files : list
            List of forcefield files
        """
        # Deepcopy forcefield_files to avoid appending forcefield files multiple times.
        # For some reason, the original forcefield_files keeps appending when SetupSampler is called.
        # TODO: Is this the right way to handle this issue?
        import copy
        _forcefield_files = copy.deepcopy(self.forcefield_files)

        # 3-site water models
        if self.water_model == 'tip3p':
            _forcefield_files.append(['amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'])
        elif self.water_model == 'tip3pfb':
            _forcefield_files.append(['amber/tip3pfb_standard.xml', 'amber/tip3pfb_HFE_multivalent.xml'])
        elif self.water_model == 'spce':
            _forcefield_files.append(['amber/spce_standard.xml', 'amber/spce_HFE_multivalent.xml'])
        elif self.water_model == 'opc3':
            raise NotImplementedError('see https://github.com/choderalab/rna-espaloma/blob/main/experiment/nucleoside/script/create_system_espaloma.py#L366')
        # 4-site water models
        elif self.water_model == 'tip4pew':
            _forcefield_files.append(['amber/tip4pew_standard.xml', 'amber/tip4pew_HFE_multivalent.xml'])
        elif self.water_model == 'tip4pfb':
            _forcefield_files.append(['amber/tip4pfb_standard.xml', 'amber/tip4pfb_HFE_multivalent.xml'])
        elif self.water_model == 'opc':
            _forcefield_files.append(['amber/opc_standard.xml'])
        else:
            raise NotImplementedError(f'Water model {self.water_model} is not supported.')
        # Flatten list
        new_forcefield_files = []
        for f in _forcefield_files:
            if isinstance(f, list):
                new_forcefield_files.extend(f)
            else:
                new_forcefield_files.append(f)

        #return new_forcefield_files
        self.forcefield_files = new_forcefield_files


    def _get_water_class(self):
        """Get water class (3-site or 4-site) based on the water model.

        Set self.water_class to tip3p or tip4pew based on the water model.

        3-site water models: tip3p, tip3pfb, spce
        4-site water models: tip4pew, tip4pfb, opc
        """
        if self.water_model in ['tip3p', 'tip3pfb', 'spce']:
            self.water_class = 'tip3p'
        elif self.water_model in ['tip4pew', 'tip4pfb', 'opc']:
            self.water_class = 'tip4pew'
        else:
            raise NotImplementedError(f'Water model {self.water_model} is not supported.')


    def _load_biopolymer_file(self):
        """Load biopolymer file."""
        with open(self._biopolymer_file, 'r') as f:
            self._biopolymer = app.PDBFile(f)  # biopolymer.positions is openmm.unit.quantity.Quantity
            # TODO: is this necessary?
            if self._biopolymer.positions.unit != unit.nanometers:
                raise Warning(f"biopolymer positions unit is expected to be nanometers but got {self._biopolymer.positions.unit}")


    def _load_ligand_file(self):
        """Load ligand file."""
        from rdkit import Chem
        from openff.toolkit.topology import Molecule

        suppl = Chem.SDMolSupplier(self._ligand_file)
        mols = [ x for x in suppl ]
        mol = mols[0]
        #mol.SetProp("_Name", "MOL")   # For some reason, the ligand name is changed to UNK when simulation system is created.
        self._ligand_offmol = Molecule.from_rdkit(mol)   
        #self._ligand_offmol = Molecule.from_file(ligand_file, allow_undefined_stereo=True)   # Is this better?
        self._ligand_positions = self._ligand_offmol.conformers[0]   # ligand.position is pint.util.Quantity            
        self._ligand_positions = self._ligand_positions.to_openmm()
        self._ligand_positions = self._ligand_positions.in_units_of(unit.nanometers)
        self._ligand_topology = self._ligand_offmol.to_topology().to_openmm()
        

    def _get_complex(self):
        """Merge biopolymer and ligand topology and position. Return complex topology and position."""
        import numpy as np
        import mdtraj

        # Define complex topology and positions
        if self._biopolymer_file is not None and self._ligand_file is None:
            complex_topology = self._biopolymer.topology
            complex_positions = self._biopolymer.positions
        elif self._biopolymer_file is None and self._ligand_file is not None:
            complex_topology = self._ligand_topology
            complex_positions = self._ligand_positions
        elif self._biopolymer_file is not None and self._ligand_file is not None:
            _logger.debug("Merge biopolymer-ligand topology")
            # Convert openmm topology to mdtraj topology
            biopolymer_md_topology = mdtraj.Topology.from_openmm(self._biopolymer.topology)
            ligand_md_topology = mdtraj.Topology.from_openmm(self._ligand_topology)
            # Merge topology
            complex_md_topology = biopolymer_md_topology.join(ligand_md_topology)
            complex_topology = complex_md_topology.to_openmm()
            # Get number of atoms
            n_atoms_total = complex_topology.getNumAtoms()
            n_atoms_biopolymer = self._biopolymer.topology.getNumAtoms()
            n_atoms_ligand = self._ligand_topology.getNumAtoms()
            assert n_atoms_total == n_atoms_biopolymer + n_atoms_ligand, "Number of atoms after merging the biopolymer and ligand topology does not match"
            _logger.debug(f"Total atoms: {n_atoms_total} (biopolymer: {n_atoms_biopolymer}, ligand: {n_atoms_ligand})")
            # Convert ligand position units to nanometers before combining the positions
            # Note: Do we need to ensure the units to be the same? Or will it automatically convert to nanometers if the units are different?
            complex_positions = unit.Quantity(np.zeros([n_atoms_total, 3]), unit=unit.nanometers)
            complex_positions[:n_atoms_biopolymer, :] = self._biopolymer.positions
            complex_positions[n_atoms_biopolymer:n_atoms_biopolymer+n_atoms_ligand, :] = self._ligand_positions
       
        return complex_topology, complex_positions
    
    
    def create_system(self, biopolymer_file=None, ligand_file=None):
        """Create biopolymer-ligand system and export serialized system XML file and solvated pdb file.

        Parameters
        ----------
        biopolymer_file : str
            biopolymer pdb file

        ligand_file : str
            ligand sdf file. The first ligand entry will be used if multiple ligands are stored.

        Returns
        -------
        None
        """
        from openmmforcefields.generators import SystemGenerator
        from openmm import MonteCarloBarostat
        from openmm import LangevinMiddleIntegrator

        # Load biopolymer and ligand files
        self._biopolymer_file = biopolymer_file
        self._ligand_file = ligand_file

        if self._biopolymer_file is None and self._ligand_file is None:
            raise ValueError("At least one biopolymer (.pdb) or ligand (.sdf) file must be provided")
        if self._biopolymer_file is not None:
            ext = os.path.splitext(self._biopolymer_file)[-1].lower()
            assert ext == '.pdb', f'Biopolymer file format must be PDB but got {ext}'
            self._load_biopolymer_file()
        if self._ligand_file is not None:
            ext = os.path.splitext(self._ligand_file)[-1].lower()
            assert ext == '.sdf', f'Ligand file format must be SDF but got {ext}'
            self._load_ligand_file()
        
        # Merge topology and position and get complex topology and position
        self._complex_topology, self._complex_positions = self._get_complex()

        # Initialize system generator.
        _logger.debug("Initialize system generator")
        forcefield_kwargs = {'removeCMMotion': True, 'ewaldErrorTolerance': self.pme_tol, 'constraints' : app.HBonds, 'rigidWater': True, 'hydrogenMass' : self.hmass}
        periodic_forcefield_kwargs = {'nonbondedMethod': app.PME}
        barostat = MonteCarloBarostat(self.pressure, self.temperature, self.barostat_period)

        # SystemGenerator will automatically load the TemplateGenerator based on the given `small_molecule_forcefield`.
        # Available TemplateGenerator: [ GAFFTemplateGenerator, SMIRNOFFTemplateGenerator, EspalomaTemplateGenerator ]
        # Template generator kwargs is only valid for EspalomaTemplateGenerator and will be ignore by GAFFTemplateGenerator and SMIRNOFFTemplateGenerator.
        template_generator_kwargs = None
        if "espaloma" in self.small_molecule_forcefield:
            # Hard coded for now. We will use the default settings for EspalomaTemplateGenerator.
            template_generator_kwargs = {"reference_forcefield": "openff_unconstrained-2.0.0", "charge_method": "nn"}
        
        # TODO: How can I set the ligand residue name to an arbitrary name?
        self._system_generator = SystemGenerator(forcefields=self.forcefield_files, forcefield_kwargs=forcefield_kwargs, 
                                                 periodic_forcefield_kwargs = periodic_forcefield_kwargs, barostat=barostat, 
                                                 small_molecule_forcefield=self.small_molecule_forcefield, cache=None, 
                                                 template_generator_kwargs=template_generator_kwargs)
        
        if ligand_file is not None:
            _logger.debug("Add molecules to system generator")
            self._system_generator.template_generator.add_molecules(self._ligand_offmol)
            
        # Solvate system
        _logger.debug("Solvating system...")
        modeller = app.Modeller(self._complex_topology, self._complex_positions)
        modeller.addSolvent(self._system_generator.forcefield, model=self.water_class, padding=self.solvent_padding, ionicStrength=self.ionic_strength)

        # Create system
        self.modeller_solvated_topology = modeller.getTopology()
        self.modeller_solvated_positions = modeller.getPositions()
        self.modeller_solvated_system = self._system_generator.create_system(self.modeller_solvated_topology)

        # Regenerate system if espaloma is used
        if "espaloma" in self.small_molecule_forcefield and self.override_with_espaloma == True:            
            #
            # Note: We will delete the original `self._system_generator` and create a new one to regenerate the system with espaloma.
            #
            # The same `self._system_generator` should be able to regenerate the system with espaloma but it failed for the RNA test system 
            # (espfit/data/target/testsystems/nucleoside/pdbfixer_min.pdb).
            # No explicit error message was given. It failed to show the following logging information:
            #
            # _logger.debug(f'Requested to generate parameters for residue {residue}')
            # https://github.com/openmm/openmmforcefields/blob/main/openmmforcefields/generators/template_generators.py#L285
            #
            # However, it works for protein test systems (espfit/data/target/testsystems/protein-ligand/target.pdb).
            #
            # As a workaround, we will delete the original `self._system_generator` and create a new one to regenerate the system with espaloma.
            # Only water and ion forcefield files will be used to regenerate the system. Solute molecules will be parametrized with espaloma.
            # 
            _logger.info("Regenerate system with espaloma")

            # Re-create system generator
            del self._system_generator
            #self.forcefield_files = self._update_forcefield_files(forcefield_files=[])  # Get water and ion forcefield files
            self.forcefield_files = []    # Initialize forcefield_files to empty list and get water and ion forcefield files
            self._update_forcefield_files()
            self._system_generator = SystemGenerator(
                forcefields=self.forcefield_files, forcefield_kwargs=forcefield_kwargs, periodic_forcefield_kwargs = periodic_forcefield_kwargs, barostat=barostat, 
                small_molecule_forcefield=self.small_molecule_forcefield, cache=None, template_generator_kwargs=template_generator_kwargs)
            
            # Regenerate system with espaloma
            self.new_solvated_system, self.new_solvated_topology = self._regenerate_espaloma_system()
        else:
            self.new_solvated_system = self.modeller_solvated_system
            self.new_solvated_topology = self.modeller_solvated_topology

        # Save solvated pdb file
        outfile = os.path.join(self.output_directory_path, f"solvated.pdb")
        with open(f"{outfile}", "w") as wf:
            app.PDBFile.writeFile(self.new_solvated_topology, self.modeller_solvated_positions, file=wf, keepIds=True)

        # Create simulation
        self.integrator = LangevinMiddleIntegrator(self.temperature, 1/unit.picosecond, self.timestep)
        self.simulation = app.Simulation(self.new_solvated_topology, self.new_solvated_system, self.integrator, self.platform)
        self.simulation.context.setPositions(self.modeller_solvated_positions)


    def _regenerate_espaloma_system(self):
        """Regenerate system with espaloma. Parameterization of biopolymer and ligand self-consistently.

        Reference
        ---------
        * https://github.com/kntkb/perses/blob/support-protein-espaloma/perses/app/relative_setup.py#L883
        * https://github.com/openforcefield/proteinbenchmark/blob/main/proteinbenchmark/system_setup.py#L651

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        import mdtraj
        from openff.toolkit import Molecule

        # Check biopolymer chains
        mdtop = mdtraj.Topology.from_openmm(self.modeller_solvated_topology)
        chain_indices = [ chain.index for chain in self.modeller_solvated_topology.chains() ]
        biopolymer_chain_indices = [ chain_index for chain_index in chain_indices if mdtop.select(f"not (water or resname NA or resname K or resname CL or resname UNK) and chainid == {chain_index}").any() ]
        _logger.debug(f"Biopolymer chain indices: {biopolymer_chain_indices}")

        # Get OpenMM topology of solute with one residue per molecule. 
        # Espaloma will use residue name "XX". Check conflicting residue names.
        conflict_resnames = [ residue.name for residue in mdtop.residues if residue.name.startswith("XX") ]
        if conflict_resnames:
            raise Exception('Found conflict residue name in biopolymer.')

        # Initilize espaloma topology
        # TODO: Should this be `self.new_solvated_topology` or `new_solvated_topology`?
        self.new_solvated_topology = app.Topology()
        self.new_solvated_topology.setPeriodicBoxVectors(self.modeller_solvated_topology.getPeriodicBoxVectors())
        new_atoms = {}

        # Regenerate biopolymer topology
        chain_index = 0
        _logger.debug(f"Regenerating biopolymer topology...")
        for chain in self.modeller_solvated_topology.chains():
            new_chain = self.new_solvated_topology.addChain(chain.id)
            # Convert biopolymer into a single residue
            if chain.index in biopolymer_chain_indices:
                resname = f'XX{chain_index:01d}'
                resid = '1'
                chain_index += 1
                new_residue = self.new_solvated_topology.addResidue(resname, new_chain, resid)
            for residue in chain.residues():
                if residue.chain.index not in biopolymer_chain_indices:
                    new_residue = self.new_solvated_topology.addResidue(residue.name, new_chain, residue.id)
                for atom in residue.atoms():
                    new_atom = self.new_solvated_topology.addAtom(atom.name, atom.element, new_residue, atom.id)
                    new_atoms[atom] = new_atom
        # Regenerate bond information
        for bond in self.modeller_solvated_topology.bonds():
            if bond[0] in new_atoms and bond[1] in new_atoms:
                self.new_solvated_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]])
        
        # Add molecules to template generator (EspalomaTemplateGenerator).
        if self._ligand_file is not None:
            self._system_generator.template_generator.add_molecules(self._ligand_offmol)
        if self._biopolymer_file is not None and len(biopolymer_chain_indices) == 1:
            openff_molecule = Molecule.from_file(self._biopolymer_file)
            openff_topology = openff_molecule.to_topology()
            self._system_generator.template_generator.add_molecules(openff_topology.unique_molecules)
        elif self._biopolymer_file is not None and len(biopolymer_chain_indices) > 1:
            # Note: This is a temporary workaround to support multiple biopolymer chains.
            # `template_generator.add_molecules(openff_topology.unique_molecules)` cannot handle 
            # multiple biopolymer chains for some reason.
                        
            import glob
            # Save complexed model and seperate biopolymers into indivdual pdb files according to chain ID
            complex_espaloma_filename = os.path.join(self.output_directory_path, "complex_solvated_espaloma.pdb")
            if not os.path.exists(complex_espaloma_filename):
                with open(complex_espaloma_filename, 'w') as outfile:
                    app.PDBFile.writeFile(self.new_solvated_topology, self.modeller_solvated_positions, outfile)
            if not biopolymer_espaloma_filenames:
                for chain_index in biopolymer_chain_indices:
                    t = mdtraj.load_pdb(complex_espaloma_filename)
                    indices = t.topology.select(f"chainid == {chain_index}")
                    t.atom_slice(indices).save_pdb(os.path.join(self.output_directory_path, f"biopolymer_espaloma_{chain_index}.pdb"))
                biopolymer_espaloma_filenames = glob.glob(self.output_directory_path + "/biopolymer_espaloma_*.pdb")
            biopolymer_molecules = [ Molecule.from_file(biopolymer_filename) for biopolymer_filename in biopolymer_espaloma_filenames ]
            self._system_generator.template_generator.add_molecules(biopolymer_molecules)

        # Regenerate system with system generator
        self.new_solvated_system = self._system_generator.create_system(self.new_solvated_topology)
        self.new_solvated_topology = self._update_espaloma_topology()

        return self.new_solvated_system, self.new_solvated_topology


    def _update_espaloma_topology(self):
        """Update espaloma topology to reflect the original residue names.

        This method updates the topology to reflect the changes made to the system.
        It converts the residue names of the solute molecules parameterized with espaloma
        to their original names.

        Returns
        -------
        app.Topology : The updated topology reflecting the new system.
        """
        _logger.debug("Update residue names in espaloma topology.")
        
        # Get original residue names.
        atom_name_lookup = []
        for residue in self.modeller_solvated_topology.residues():
            # Assume that solute molecules parameterized with espaloma comes before water molecules.
            if residue.name == 'HOH': break
            a = [ {"name": atom.name, "index": atom.index, "resname": atom.residue.name, "resid": atom.residue.id } for atom in residue.atoms() ][0]
            atom_name_lookup.append(a)

        # Create new topology and convert residue names to its original names.
        new_topology = app.Topology()
        new_topology.setPeriodicBoxVectors(self.modeller_solvated_topology.getPeriodicBoxVectors())
        new_atoms = {}

        i = 0
        for chain in self.new_solvated_topology.chains():    
            new_chain = new_topology.addChain(chain.id)
            for residue in chain.residues():
                # Covert all residues parameterized with espaloma to its original residue names.
                if residue.name.startswith('XX'):
                    for atom in residue.atoms():
                        try:
                            if atom_name_lookup[i]['name'] == atom.name and atom_name_lookup[i]['index'] == atom.index:
                                resname = atom_name_lookup[i]['resname']
                                resid = atom_name_lookup[i]['resid']
                                newResidue = new_topology.addResidue(resname, new_chain, resid, residue.insertionCode)
                                i += 1
                        except:
                            pass
                        new_atom = new_topology.addAtom(atom.name, atom.element, newResidue, atom.id)
                        new_atoms[atom] = new_atom
                else:
                    # Just copy the residue over.
                    newResidue = new_topology.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
                    for atom in residue.atoms():
                        new_atom = new_topology.addAtom(atom.name, atom.element, newResidue, atom.id)
                        new_atoms[atom] = new_atom     
        for bond in self.new_solvated_topology.bonds():
            if bond[0] in new_atoms and bond[1] in new_atoms:
                new_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]])

        return new_topology


    @classmethod
    def from_xml(cls, input_directory_path=None):
        """Load serialized system XML file and solvated pdb file.

        Parameters
        ----------
        input_directory_path : str, optional
            Input directory path to restart simulation. Default is None.

        Returns
        -------
        instance : object
        """
        instance = cls()
        
        if input_directory_path is not None:
            instance.input_directory_path = input_directory_path

        from openmm import XmlSerializer
        
        # Deserialize system file and load system
        with open(os.path.join(instance.input_directory_path, 'system.xml'), 'r') as f:
            system = XmlSerializer.deserialize(f.read())

        # Deserialize integrator file and load integrator
        with open(os.path.join(instance.input_directory_path, 'integrator.xml'), 'r') as f:
            integrator = XmlSerializer.deserialize(f.read())

        # Set up simulation
        pdb = app.PDBFile(os.path.join(instance.input_directory_path, 'state.pdb'))
        instance.simulation = app.Simulation(pdb.topology, system, integrator, instance.platform)

        # Load state
        with open(os.path.join(instance.input_directory_path, 'state.xml'), 'r') as f:
            state = XmlSerializer.deserialize(f.read())
        instance.simulation.context.setState(state)
    
        return instance
    