
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce




def process_single_transition_ranking(ranking_df, human_tfs_names, known_tfs_list):
    """
    Process a single transition ranking by adding TF annotations
    
    Parameters:
    -----------
    ranking_df : pd.DataFrame
        Gene ranking dataframe with 'all' column
    human_tfs_names : list
        List of all human transcription factor names
    known_tfs_list : list
        List of known TFs for this specific transition
        
    Returns:
    --------
    processed_ranking : pd.DataFrame
        Processed ranking with TF and known_TF columns
    """
    # Create a copy to avoid modifying original
    processed_ranking = ranking_df.copy()
    
    # Add TF annotation
    processed_ranking["TF"] = [
        gene in human_tfs_names for gene in processed_ranking["all"]
    ]
    
    # Filter to only TFs and create explicit copy to avoid SettingWithCopyWarning
    processed_ranking = processed_ranking.query("TF == True").copy()
    
    # Add known TF annotation
    processed_ranking.loc[:, "known_TF"] = [
        gene in known_tfs_list for gene in processed_ranking["all"]
    ]
    
    return processed_ranking


def assign_tf_ranks(transition_graph, transition_name, tfs_list, 
                   human_tfs_names, tfs_key="TFs", tfs_rank_key="TFs_rank"):
    """
    Assign TF ranks to a specific transition in the transition graph
    
    Parameters:
    -----------
    transition_graph : dict
        Dictionary containing all transition results
    transition_name : str
        Name of the transition (e.g., "HSC->Meg")
    tfs_list : list
        List of TFs to rank for this transition
    human_tfs_names : list
        List of all human transcription factor names
    tfs_key : str
        Key name for storing TF list
    tfs_rank_key : str
        Key name for storing TF ranks
    """
    ranking = transition_graph[transition_name]["ranking"]
    ranking["TF"] = [gene in human_tfs_names for gene in ranking["all"]]
    
    # Get TF-only ranking
    tf_ranking = ranking.query("TF == True")
    all_ranked_tfs = list(tf_ranking["all"])
    
    # Store TFs and their ranks
    transition_graph[transition_name][tfs_key] = tfs_list
    transition_graph[transition_name][tfs_rank_key] = [
        all_ranked_tfs.index(tf) if tf in all_ranked_tfs else -1 
        for tf in tfs_list
    ]


def process_all_transition_rankings(transition_graph, human_tfs_names, known_tfs_dict=None):
    """
    Process all transitions in the transition graph to extract TF rankings
    
    Parameters:
    -----------
    transition_graph : dict
        Dictionary containing all transition results
    human_tfs_names : list
        List of all human transcription factor names
    known_tfs_dict : dict, optional
        Dictionary mapping transition names to known TF lists
        If None, uses KNOWN_TFS_DICT
        
    Returns:
    --------
    processed_rankings : dict
        Dictionary of processed rankings for each transition
    """
    if known_tfs_dict is None:
        #known_tfs_dict = KNOWN_TFS_DICT
        print("No known TFs dictionary provided, skipping transition ranking analysis")
        return None
    
    processed_rankings = {}
    
    # Process standard transitions
    for transition_name, known_tfs in known_tfs_dict.items():
        if transition_name == "Ery->Neu_alt":
            continue  # Handle this separately
            
        if transition_name in transition_graph:
            ranking = transition_graph[transition_name]["ranking"]
            processed_rankings[transition_name] = process_single_transition_ranking(
                ranking, human_tfs_names, known_tfs
            )
            
            # Assign TF ranks
            assign_tf_ranks(
                transition_graph, transition_name, known_tfs, human_tfs_names
            )
    
    # Handle special case for Ery->Neu with two TF sets
    if "Ery->Neu" in transition_graph:
        # First set (LSD1, RUNX1)
        assign_tf_ranks(
            transition_graph, "Ery->Neu", known_tfs_dict["Ery->Neu"], 
            human_tfs_names, tfs_key="TFs1", tfs_rank_key="TFs_rank1"
        )
        
        # Second set (CEBPA, CEBPB, CEBPE, SPI1) 
        assign_tf_ranks(
            transition_graph, "Ery->Neu", known_tfs_dict["Ery->Neu_alt"], 
            human_tfs_names, tfs_key="TFs2", tfs_rank_key="TFs_rank2"
        )
    
    return processed_rankings


def create_reprogramming_matrix(transition_graph, 
                               transitions_config,
                               transition_pmids=None, 
                               transition_types=None,
                               total_tf_count=133):
    """
    Create reprogramming matrix from processed transition graph
    
    Parameters:
    -----------
    transition_graph : dict
        Dictionary containing all transition results with TF rankings
    transitions_config : dict
        Configuration for transitions, format:
        {
            "standard": ["HSC->Meg", "HSC->Ery", ...],  # Standard transitions
            "special": {  # Special multi-TF transitions
                "Ery->Neu": {
                    "sets": [("TFs1", "TFs_rank1", "Ery->Neu1"), 
                             ("TFs2", "TFs_rank2", "Ery->Neu2")]
                }
            }
        }
    transition_pmids : dict, optional
        Dictionary mapping transition names to PMID references
    transition_types : dict, optional
        Dictionary mapping transition names to transition types
    total_tf_count : int, default=133
        Total number of TFs for rank normalization
        
    Returns:
    --------
    reprogramming_dict : dict
        Dictionary containing reprogramming information
    reprogramming_df : pd.DataFrame
        Flattened dataframe for analysis and plotting
    """
    if transition_pmids is None:
        #transition_pmids = TRANSITION_PMIDS
        print("No transition PMIDs provided, skipping transition PMIDs analysis")
    if transition_types is None:
        #transition_types = TRANSITION_TYPES
        print("No transition types provided, skipping transition types analysis")
    
    reprogramming_dict = {}
    
    # Process standard transitions
    standard_transitions = transitions_config.get("standard", [])
    for transition in standard_transitions:
        if transition in transition_graph:
            reprogramming_dict[transition] = {
                "genes": transition_graph[transition]["TFs"],
                "rank": transition_graph[transition]["TFs_rank"],
                "PMID": transition_pmids.get(transition, None)
            }
    
    # Process special multi-TF transitions
    special_transitions = transitions_config.get("special", {})
    for base_transition, config in special_transitions.items():
        if base_transition in transition_graph:
            for tf_key, rank_key, output_name in config["sets"]:
                reprogramming_dict[output_name] = {
                    "genes": transition_graph[base_transition][tf_key],
                    "rank": transition_graph[base_transition][rank_key],
                    "PMID": transition_pmids.get(output_name, None)
                }
    
    # Create flattened dataframe
    reprogramming_df = pd.DataFrame(reprogramming_dict)
    
    # Validate data consistency
    for key in reprogramming_df.columns:
        genes_len = len(reprogramming_df[key]["genes"])
        rank_len = len(reprogramming_df[key]["rank"])
        assert genes_len == rank_len, f"Length mismatch in {key}: genes={genes_len}, rank={rank_len}"
    
    # Flatten the data
    all_genes = reduce(lambda a, b: a + b, reprogramming_df.loc["genes", :])
    all_rank = reduce(lambda a, b: a + b, reprogramming_df.loc["rank", :])
    all_keys = np.repeat(
        np.array(list(reprogramming_dict.keys())), 
        [len(genes) for genes in reprogramming_df.loc["genes", :]]
    )
    
    # Create final dataframe - explicitly create as independent DataFrame
    final_df = pd.DataFrame({
        "genes": all_genes, 
        "rank": all_rank, 
        "transition": all_keys
    }).copy()  # Ensure it's a copy
    
    # Filter out unranked genes and create explicit copy
    final_df = final_df.query("rank > -1").copy()
    
    # Add transition type using .loc to avoid warnings
    final_df.loc[:, "type"] = final_df["transition"].map(transition_types)
    
    # Normalize ranks using .loc to avoid warnings
    final_df.loc[:, "rank"] = final_df["rank"] / total_tf_count
    final_df.loc[:, "rank"] = 1 - final_df["rank"]  # Higher rank = higher score
    
    return reprogramming_dict, final_df


def plot_transition_tf_analysis(reprogramming_df, transition_type="transdifferentiation", 
                               figsize=(8, 5), score_threshold=0.8,
                               transition_color_dict=None):
    """
    Plot TF analysis results for specific transition types
    
    Parameters:
    -----------
    reprogramming_df : pd.DataFrame
        Dataframe containing reprogramming information
    transition_type : str
        Type of transition to plot ("development", "reprogramming", "transdifferentiation")
    figsize : tuple
        Figure size
    score_threshold : float
        Vertical line position for score threshold
    transition_color_dict : dict, optional
        Custom color mapping for transition types
    """
    # Default color dictionary
    if transition_color_dict is None:
        transition_color_dict = {
            "development": "#2E3192", 
            "reprogramming": "#EC2227", 
            "transdifferentiation": "#B9519E"
        }
    
    # Filter data
    subset_df = reprogramming_df.query(f"type == '{transition_type}'")
    
    if subset_df.empty:
        print(f"No data found for transition type: {transition_type}")
        return
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Scatter plot
    sns.scatterplot(
        y="transition", x="rank", data=subset_df,
        ec=None, hue="type", alpha=0.8, ax=ax, s=50,
        palette=transition_color_dict, clip_on=False
    )
    
    
    
    from adjustText import adjust_text
    texts=[]
    for idx, row in subset_df.iterrows():
        if row['rank']<=0.6: 
            pass
        else:
            texts.append(ax.text(row["rank"],
                                row["transition"],
                                row["genes"],
                                fontdict={'size':9,'color':'black'}
                                ))
    
    adjust_text(texts,
                only_move={"text": "y", "static": "xy", "explode": "xy", "pull": "xy"},
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Add threshold line
    plt.axvline(score_threshold, linestyle="--", lw=0.5)
    
    # Formatting
    ax.set_xlim(0.6, 1.01)
    ax.set_xlabel("Score", fontsize=14)
    ax.set_ylabel("Transition", fontsize=14)
    ax.legend().set_visible(False)
    
    # Spine formatting
    ax.spines.top.set_position(("outward", 10))
    ax.spines.bottom.set_position(("outward", 10))
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    
    plt.tick_params(axis='both', which='both', labelsize=10)
    plt.tight_layout()
    return fig, ax


def analyze_transition_tfs(transition_graph, human_tfs_names, 
                          transitions_config,
                          plot_type="transdifferentiation", 
                          known_tfs_dict=None, 
                          transition_pmids=None, 
                          transition_types=None,
                          total_tf_count=133,
                          transition_color_dict=None,
                          figsize=(8, 5)):
    """
    Complete pipeline for analyzing transition TFs
    
    Parameters:
    -----------
    transition_graph : dict
        Dictionary containing all transition results
    human_tfs_names : list
        List of all human transcription factor names
    transitions_config : dict
        Configuration for transitions processing
    plot_type : str
        Type of transition to plot
    known_tfs_dict : dict, optional
        Custom known TFs dictionary
    transition_pmids : dict, optional
        Custom PMID dictionary
    transition_types : dict, optional
        Custom transition types dictionary
    total_tf_count : int, default=133
        Total number of TFs for rank normalization
    transition_color_dict : dict, optional
        Custom color mapping for transition types
        
    Returns:
    --------
    tuple
        (processed_rankings, reprogramming_dict, reprogramming_df)
    """
    print("Processing transition rankings...")
    processed_rankings = process_all_transition_rankings(
        transition_graph, human_tfs_names, known_tfs_dict
    )
    
    print("Creating reprogramming matrix...")
    reprogramming_dict, reprogramming_df = create_reprogramming_matrix(
        transition_graph, transitions_config, transition_pmids, 
        transition_types, total_tf_count
    )
    
    print(f"Plotting {plot_type} transitions...")
    plot_transition_tf_analysis(reprogramming_df, plot_type, 
                               transition_color_dict=transition_color_dict,
                               figsize=figsize)
    
    return processed_rankings, reprogramming_dict, reprogramming_df


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def consolidate_processed_rankings(processed_rankings, transitions_to_include=None):
    """
    Consolidate processed rankings from the transition analysis results
    
    Parameters:
    -----------
    processed_rankings : dict
        Dictionary of processed rankings from process_all_transition_rankings()
    transitions_to_include : list, optional
        List of specific transitions to include. If None, includes all available transitions
        
    Returns:
    --------
    consolidated_df : pd.DataFrame
        Consolidated ranking DataFrame with source information
    """
    if not processed_rankings:
        raise ValueError("Processed rankings dictionary cannot be empty")
    
    rankings_list = []
    
    # Get transitions to process
    if transitions_to_include is None:
        transitions_to_include = list(processed_rankings.keys())
    
    # Extract available transitions that exist in processed_rankings
    available_transitions = [t for t in transitions_to_include if t in processed_rankings]
    
    if not available_transitions:
        raise ValueError(f"None of the specified transitions found in processed_rankings. "
                        f"Available: {list(processed_rankings.keys())}")
    
    print(f"Consolidating {len(available_transitions)} transitions: {available_transitions}")
    
    # Consolidate rankings
    for transition_name in available_transitions:
        df = processed_rankings[transition_name].copy()
        df['source'] = transition_name
        rankings_list.append(df)
    
    # Concatenate all rankings
    consolidated_df = pd.concat(rankings_list, ignore_index=True)
    
    return consolidated_df


def calculate_priority_scores_from_consolidated(consolidated_df):
    """
    Calculate priority scores for consolidated rankings from processed results
    
    Parameters:
    -----------
    consolidated_df : pd.DataFrame
        Consolidated ranking DataFrame from consolidate_processed_rankings()
        
    Returns:
    --------
    df_with_scores : pd.DataFrame
        DataFrame with added priority_score column
    """
    df_with_scores = consolidated_df.copy()
    
    # Get unique sources to determine reference size
    sources = df_with_scores['source'].unique()
    n_sources = len(sources)
    
    # Calculate reference size (assume all sources have same number of rows)
    reference_size = len(df_with_scores) // n_sources
    
    print(f"Calculating priority scores with reference_size={reference_size} for {n_sources} sources")
    
    # Calculate priority scores
    df_with_scores["priority_score"] = (
        1 - np.tile(np.arange(reference_size), n_sources) / reference_size
    )
    
    return df_with_scores


def plot_roc_curve(y_true, y_scores, 
                   figsize=(4, 4), 
                   fontsize=12, 
                   linewidth=1.5,
                   roc_color="darkorange",
                   diagonal_color="navy",
                   title=None,
                   legend_size=12,
                   hide_zero_ticks=True,
                   return_fig=False):
    """
    Plot ROC curve for binary classification performance
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Predicted scores/probabilities
    figsize : tuple, default=(10, 10)
        Figure size
    fontsize : int, default=16
        Font size for labels
    linewidth : float, default=1.5
        Line width for curves
    roc_color : str, default="darkorange"
        Color for ROC curve
    diagonal_color : str, default="navy"
        Color for diagonal reference line
    title : str, optional
        Plot title
    legend_size : int, default=20
        Legend font size
    hide_zero_ticks : bool, default=True
        Whether to hide zero ticks on axes
        
    Returns:
    --------
    fpr, tpr, roc_auc : tuple
        False positive rate, true positive rate, and AUC score
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.tick_params(axis='both', which='both', labelsize=fontsize)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color=roc_color, lw=linewidth, 
             label="ROC curve (area = %0.2f)" % roc_auc)
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color=diagonal_color, lw=linewidth, linestyle="--")
    
    # Set limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontsize=fontsize)
    
    # Hide zero ticks if requested
    if hide_zero_ticks:
        ax = plt.gca()
        plt.setp(ax.get_yticklabels()[0], visible=False)    
        plt.setp(ax.get_xticklabels()[0], visible=False)
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=fontsize)
    
    # Add legend
    plt.legend(loc="lower right", prop={'size': legend_size})
    
    plt.tight_layout()
    
    if return_fig:
        return ax
    else:
        return fpr, tpr, roc_auc


def analyze_tf_roc_performance(processed_rankings, 
                              transitions_to_include=None,
                              plot_roc=True,
                              roc_plot_params=None):
    """
    Analyze TF ranking performance using ROC curve based on processed rankings
    
    Parameters:
    -----------
    processed_rankings : dict
        Dictionary of processed rankings from process_all_transition_rankings()
    transitions_to_include : list, optional
        List of specific transitions to include in analysis
    plot_roc : bool, default=True
        Whether to plot ROC curve
    roc_plot_params : dict, optional
        Parameters for ROC plot customization
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'consolidated_df': Consolidated DataFrame with priority scores
        - 'fpr': False positive rate
        - 'tpr': True positive rate  
        - 'roc_auc': AUC score
        - 'performance_summary': Summary statistics
    """
    # Default ROC plot parameters
    if roc_plot_params is None:
        roc_plot_params = {}
    
    print("=== TF ROC Performance Analysis ===")
    
    # Step 1: Consolidate processed rankings
    print("Consolidating processed rankings...")
    consolidated_df = consolidate_processed_rankings(processed_rankings, transitions_to_include)
    
    # Step 2: Calculate priority scores
    print("Calculating priority scores...")
    df_with_scores = calculate_priority_scores_from_consolidated(consolidated_df)
    
    # Step 3: Prepare data for ROC analysis
    if 'known_TF' not in df_with_scores.columns:
        raise ValueError("DataFrame must contain 'known_TF' column for performance analysis")
    
    y_true = df_with_scores["known_TF"].astype(int)
    y_scores = df_with_scores["priority_score"]
    
    print("Calculating ROC metrics...")
    # Step 4: Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Step 5: Plot ROC curve if requested
    if plot_roc:
        print("Plotting ROC curve...")
        plot_roc_curve(y_true, y_scores, **roc_plot_params)
    
    # Step 6: Generate performance summary
    performance_summary = {
        'total_predictions': len(y_true),
        'known_tfs': int(y_true.sum()),
        'unknown_tfs': int(len(y_true) - y_true.sum()),
        'roc_auc': float(roc_auc),
        'mean_priority_score_known': float(y_scores[y_true == 1].mean()),
        'mean_priority_score_unknown': float(y_scores[y_true == 0].mean()),
        'transitions_analyzed': list(consolidated_df['source'].unique())
    }
    
    print(f"\nPerformance Analysis Summary:")
    print(f"- ROC AUC: {roc_auc:.3f}")
    print(f"- Total predictions: {performance_summary['total_predictions']}")
    print(f"- Known TFs: {performance_summary['known_tfs']}")
    print(f"- Unknown TFs: {performance_summary['unknown_tfs']}")
    print(f"- Mean priority score (known TFs): {performance_summary['mean_priority_score_known']:.3f}")
    print(f"- Mean priority score (unknown TFs): {performance_summary['mean_priority_score_unknown']:.3f}")
    print(f"- Transitions analyzed: {performance_summary['transitions_analyzed']}")
    
    return {
        'consolidated_df': df_with_scores,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'performance_summary': performance_summary
    }


def get_tf_statistics(processed_rankings, reprogramming_df):
    """
    Get TF statistics from processed rankings and reprogramming analysis
    
    Parameters:
    -----------
    processed_rankings : dict
        Dictionary of processed rankings
    reprogramming_df : pd.DataFrame
        Reprogramming DataFrame with gene information
        
    Returns:
    --------
    tf_stats : dict
        Dictionary containing TF statistics
    """
    # Get all TFs from processed rankings
    all_tfs_sets = []
    for transition, ranking_df in processed_rankings.items():
        if 'all' in ranking_df.columns:
            tfs_in_transition = set(ranking_df['all'].values)
            all_tfs_sets.append(tfs_in_transition)
    
    # Combine all unique TFs
    all_tfs = set()
    for tf_set in all_tfs_sets:
        all_tfs.update(tf_set)
    
    # Get valid TFs from reprogramming analysis
    if 'genes' in reprogramming_df.columns:
        valid_tfs = set(reprogramming_df["genes"].values)
    else:
        valid_tfs = set()
    
    # Calculate overlap
    overlap = all_tfs & valid_tfs
    
    return {
        'all_tfs': list(all_tfs),
        'valid_tfs': list(valid_tfs),
        'overlap_tfs': list(overlap),
        'n_all_tfs': len(all_tfs),
        'n_valid_tfs': len(valid_tfs),
        'n_overlap': len(overlap),
        'overlap_percentage': (len(overlap) / len(all_tfs) * 100) if all_tfs else 0
    }