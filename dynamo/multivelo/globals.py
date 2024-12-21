import os
import platform

# Determine platform on which analysis is running
running_on = platform.system()

# Set up locale configuration here
REPO_PATH, ROOT_PATH = None, None # To make the lint checker happy ...
if running_on == 'Darwin':
    # ... OSX system
    # ... ... root path
    ROOT_PATH = '/Users/cordessf/OneDrive'         # <============= CHANGE THIS !!!

    # ... ... repo path
    REPO_PATH = os.path.join(ROOT_PATH, 'ACI', 'Repositories')
elif running_on == 'Linux':
    # ... Linux system
    # ... ... root path
    ROOT_PATH = '/data/LIRGE'                      # <============= CHANGE THIS !!!

    # ... ... repo path
    REPO_PATH = os.path.join(ROOT_PATH, 'Repositories')

# ... Path to base directory (where code and results are kept)
BASE_PATH = os.path.join(REPO_PATH, 'MultiDynamo')

# ... Path to cache intermediate results
CACHE_PATH = os.path.join(ROOT_PATH, 'cache')
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# ... Path to data
DATA_PATH = os.path.join(ROOT_PATH, 'external_data', 'multiome')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# ... Path to reference data
REFERENCE_DATA_PATH = os.path.join(ROOT_PATH, 'reference_data')
if not os.path.exists(REFERENCE_DATA_PATH):
    os.makedirs(REFERENCE_DATA_PATH)

# Structure the data as it would come out of a cellranger run
# ... cellranger outs directory
OUTS_PATH = os.path.join(DATA_PATH, 'outs')
if not os.path.exists(OUTS_PATH):
    os.makedirs(OUTS_PATH)

# Path to ATAC-seq data
ATAC_PATH = os.path.join(ROOT_PATH, 'external_data', '10k_human_PBMC_ATAC')

# Path to genome annotation
GTF_PATH = os.path.join(REFERENCE_DATA_PATH, 'annotation', 'Homo_sapiens.GRCh38.112.gtf.gz')

# Path to multiomic data
MULTIOME_PATH = DATA_PATH

# Path to RNA-seq data
RNA_PATH = os.path.join(ROOT_PATH, 'external_data', '10k_human_PBMC_RNA')
