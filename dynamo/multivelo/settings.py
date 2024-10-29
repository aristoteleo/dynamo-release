import os

"""Settings
"""

# the desired verbosity
global VERBOSITY

# cwd: The current working directory
global CWD

# the name of the file to which we're writing the log files
global LOG_FOLDER

# the name of the file to which we're writing the logs
# (If left to the default value of None, we don't write to a file)
global LOG_FILENAME

# the name of the gene the code is processing
global GENE

VERBOSITY = 1
CWD = os.path.abspath(os.getcwd())
LOG_FOLDER = os.path.join(CWD, "../logs")
LOG_FILENAME = None
GENE = None

