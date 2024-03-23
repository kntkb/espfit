import pytest
from importlib.resources import files
from espfit.utils.graphs import CustomGraphDataset
from espfit.app.train import EspalomaModel


@pytest.fixture
def test_create_espaloma_from_toml(tmpdir):
    """Test function to load a TOML configuration file and create an EspalomaModel object.

    Returns
    -------
    model : espfit.app.train.EspalomaModel
        The created EspalomaModel object.
    """
    filename = files('espfit').joinpath('data/config/config.toml')   # PosixPath
    model = EspalomaModel.from_toml(str(filename))
    model.output_directory_path = str(tmpdir)   # Update output directory path

    return model


@pytest.fixture
def test_load_dataset(tmpdir):
    """Test function to load a dataset and prepare it for training.

    Parameters
    ----------
    tmpdir : py._path.local.LocalPath   # IS THIS CORRECT?
        Temporary directory.

    Notes
    -----
    This function is not intended for production use. It is a minimal example for testing purposes.

    Returns
    -------
    ds : espfit.utils.graphs.CustomGraphDataset
        The loaded dataset.
    """
    # load dataset
    path = 'data/qcdata/openff-toolkit-0.10.6/dgl2/gen2-torsion-sm'
    mydata = files('espfit').joinpath(path)
    ds = CustomGraphDataset.load(str(mydata))

    # Prepare input dataset ready for training
    temporary_directory = tmpdir.mkdir('misc')
    ds.drop_duplicates(save_merged_dataset=True, dataset_name='misc', output_directory_path=str(temporary_directory))
    ds.compute_relative_energy()
    ds.reshape_conformation_size(n_confs=50)

    return ds


def test_train_sampler(test_load_dataset, test_create_espaloma_from_toml):
    """Test function to train a sampler."""
    import os
    import glob

    # Load dataset and model
    ds = test_load_dataset
    model = test_create_espaloma_from_toml

    # Set espaloma parameters
    model.dataset_train = ds
    model.epochs = 15

    # Train
    sampler_patience = 10
    # Force sampler to run after reaching sampler patience by setting neff_threshold to 1.0
    # Set debug=True to use espaloma-0.3.2.pt instead of on-the-fly intermediate espaloma model.
    # Epochs are too short for stable espaloma model.
    model.train_sampler(sampler_patience=sampler_patience, neff_threshold=1.0, sampler_weight=1, debug=True)

    # Check outputs
    n_ckpt = len(glob.glob(os.path.join(model.output_directory_path, 'ckpt*pt')))
    assert n_ckpt == int(model.epochs / model.checkpoint_frequency)

    n_adenosine_pred_yaml = len(glob.glob(os.path.join(model.output_directory_path, 'adenosine/*/pred.yaml')))
    assert n_adenosine_pred_yaml == int(model.epochs - sampler_patience)

    n_cytidine_pred_yaml = len(glob.glob(os.path.join(model.output_directory_path, 'cytidine/*/pred.yaml')))
    assert n_cytidine_pred_yaml == int(model.epochs - sampler_patience)
