"""
Submodule containing some functions for reading some file formats containing time- or
frequency-based data and creates the respective objects. Relies on the scisave package
[1].

References
----------
[1] https://git.physik.hu-berlin.de/pylab/scisave
"""

import numpy as np
from .transfer_functions import TransferFunction
from .time_data import CounterData

try:
    import scisave
except ImportError:
    has_scisave = False
else:
    has_scisave = True


def import_json(filename, as_class, silent=False, **kwargs):
    """
    Import data from a .json file saved with scisave and create a FreqData object from
    it. All of the keyworded arguments needed to construct the class inheriting from
    Freqdata(e.g. `rbw` for `SpectrumAnalyzerData`) have to be passed to the function as
    keyworded arguments.

    Parameters
    ----------
    filename : str
    as_class : CounterData or TransferFunction
    silent : bool (optional, default False)
        import is done without printing info if False
    **kwargs :
        these keyworded arguments will be passed to the constructor of the object that
        is to be created, e.g. `divide_by` for a CounterData object.

    Returns
    -------
    instance : as defined in as_class
    """
    obj = None
    if not has_scisave:
        raise ImportError(
            "scisave (https://git.physik.hu-berlin.de/pylab/scisave) is required."
        )
    data = scisave.load_measurement(filename, silent=silent)
    if as_class == CounterData:
        freqs = data["results"]["frequencies"]
        device_settings = data["device_settings"]
        obj = CounterData(freqs, **device_settings, **kwargs)
    elif as_class == TransferFunction:
        freqs = data["results"]["frequency"]
        magnitude = data["results"]["magnitude"]
        phase = data["results"]["phase"]
        obj = TransferFunction(freqs, magnitude, phase, **kwargs)
    else:
        raise TypeError("Type {} is not supported.".format(type(as_class)))
    return obj


def import_csv(filename, as_class, delimiter=",", skip_header=0, **kwargs):
    """
    Import data from a .csv and create a FreqData object from it.
    All of the keyworded arguments needed to construct the class inheriting from
    Freqdata(e.g. `rbw` for `SpectrumAnalyzerData`) have to be passed to the function as
    keyworded arguments.

    Parameters
    ----------
    filename : str
    as_class : FreqData
        FreqData or one of its subclasses
    delimiter : str (optional)
        delimiter of the file, e.g. '\t' for .tsv files
    skip_header : int, (optional)
        the number of lines to skip at the beginning of the file

    Returns
    -------
    instance : as defined in as_class
    """
    data = np.genfromtxt(
        filename,
        dtype=float,
        delimiter=delimiter,
        skip_header=skip_header,
        comments="%",
        names=["freqs", "values"],
    )
    instance = as_class(data["freqs"], data["values"], **kwargs)
    return instance
