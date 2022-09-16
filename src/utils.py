# %%
# Imports
from configparser import ConfigParser
from typing import Any


# %%
# Helper Functions
def get_config(name: str) -> dict[str, str | float]:
    """Function to read the configuration from a file

    Args:
        name (str): the name of the process that requests its configuration

    Returns:
        dict[str, str | float]: the configuration in the form of a dicitonary
    """

    # load the configuration
    config_parser = ConfigParser()
    config_parser.read("settings.config")

    # parse numberic]
    configuration = {
        key: float(value) if value.isnumeric() else value
        for key, value in dict(config_parser[name]).items()
    }

    return configuration
