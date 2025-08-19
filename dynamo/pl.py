"""Mapping Vector Field of Single Cells
"""

from .plot import *

import os
import tempfile
import warnings
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import rcParams

dynamo_logo="""

 ███                               ████████        
█████   █████    █████    █████    ███   █████      
   ██████   ██████   ██████   ████████      ████ 
  ___                           ████            ███
 |   \ _  _ _ _  __ _ _ __  ___                 ███
 | |) | || | ' \/ _` | '  \/ _ \█████           ███ 
 |___/ \_, |_||_\__,_|_|_|_\___/█████       ████  
       |__/                        ███   █████     
Tutorial: https://dynamo-release.readthedocs.io/       
                                     █████      
"""    
# Add this at the module level
_has_printed_logo = False  # Flag to ensure logo prints only once

def style(
        dpi=80,
        dpi_save: int = 150,
        transparent: bool = False,
        facecolor='white',
        fontsize=14,
        figsize=None,
        color_map=None,
        dynamo=True,
        font_path=None,
):
    """
    Set the default parameters for matplotlib figures.
    
    Arguments:
        dpi: Resolution for matplotlib figures (80)
        dpi_save: Resolution for saving figures (150)
        transparent: Whether to save figures with transparent background (False)
        facecolor: Background color for figures ('white')
        fontsize: Default font size (14)
        figsize: Figure size tuple (None)
        color_map: Default color map (None)
        dynamo: Whether to apply dynamo-specific rcParams (True)
        font_path: Path to custom font file or 'arial' for auto-download (None)
    """
    global _has_printed_logo  # Use the global flag

    from matplotlib import rcParams

    if dpi is not None:
        rcParams["figure.dpi"] = dpi
    if dpi_save is not None:
        rcParams["savefig.dpi"] = dpi_save
    if transparent is not None:
        rcParams["savefig.transparent"] = transparent
    if facecolor is not None:
        rcParams["figure.facecolor"] = facecolor
        rcParams["axes.facecolor"] = facecolor
    if dynamo:
        set_rcParams_dynamo(fontsize=fontsize, color_map=color_map)
    if figsize is not None:
        rcParams["figure.figsize"] = figsize

    # Custom font setup
    if font_path is not None:
        # Check if user wants Arial font (auto-download)
        if font_path.lower() in ['arial', 'arial.ttf'] and not font_path.endswith('.ttf'):
            try:
                # Create a persistent cache location for the Arial font
                cache_dir = tempfile.gettempdir()
                cached_arial_path = os.path.join(cache_dir, 'dynamo_arial.ttf')
                
                # Check if Arial font is already cached
                if os.path.exists(cached_arial_path):
                    print(f"Using already downloaded Arial font from: {cached_arial_path}")
                    font_path = cached_arial_path
                else:
                    print("Downloading Arial font from GitHub...")
                    try:
                        import requests
                        arial_url = "https://github.com/kavin808/arial.ttf/raw/refs/heads/master/arial.ttf"
                        
                        # Download the font
                        response = requests.get(arial_url, timeout=30)
                        response.raise_for_status()
                        
                        # Save the font to cache location
                        with open(cached_arial_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Use the cached font file
                        font_path = cached_arial_path
                        print(f"Arial font downloaded successfully to: {cached_arial_path}")
                        
                    except ImportError:
                        print("requests module not available. Please install requests to download Arial font automatically.")
                        print("Continuing with default font settings...")
                        font_path = None
                        
            except Exception as e:
                print(f"Failed to download Arial font: {e}")
                print("Continuing with default font settings...")
                font_path = None
        
        if font_path is not None:
            try:
                # 1) Create a brand-new manager
                fm.fontManager = fm.FontManager()
                
                # 2) Add your file
                fm.fontManager.addfont(font_path)
                
                # 3) Now find out what name it uses
                name = fm.FontProperties(fname=font_path).get_name()
                print(f"Registered custom font as: {name}")
                
                # 4) Point rcParams at that name
                rcParams['font.family'] = 'sans-serif'
                rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
                
            except Exception as e:
                print(f"Failed to set custom font: {e}")
                print("Continuing with default font settings...")

    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)

    # Print the logo only once
    if not _has_printed_logo:
        print(dynamo_logo)
        _has_printed_logo = True


                  


def set_rcParams_dynamo(fontsize=14, color_map=None):
    """Set matplotlib.rcParams to Scanpy defaults.

    Call this through `settings.set_figure_params`.
    """
    # figure
    rcParams["figure.figsize"] = (4, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = 0.92 * fontsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = fontsize

    # legend
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize

    # axes grid
    rcParams["axes.grid"] = True
    rcParams["grid.color"] = ".8"

    # color map
    rcParams["image.cmap"] = rcParams["image.cmap"] if color_map is None else color_map


def set_rcParams_defaults():
    """Reset `matplotlib.rcParams` to defaults."""
    rcParams.update(mpl.rcParamsDefault)
