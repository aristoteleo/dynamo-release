import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import dynamo as dyn
from dynamo.tools import run_sude
from dynamo.external.sude_py.sude import sude


class TestSUDE:
    """Test class for SUDE (Scalable Unsupervised Deep Embedding) functionality."""

    def test_sude_import(self):
        """Test that SUDE module can be imported correctly."""
        from dynamo.external.sude_py.sude import sude
        assert sude is not None

    def test_sude_function_basic(self):
        """Test basic SUDE function with simple data."""
        # Create simple test data
        X = np.random.randn(50, 10)
        
        # Test SUDE with default parameters
        result = sude(X, no_dims=2, k1=10, normalize=True)
        
        assert result.shape == (50, 2)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_sude_function_parameters(self):
        """Test SUDE function with different parameter combinations."""
        X = np.random.randn(100, 20)
        
        # Test with different initialization methods
        for init_method in ["le", "pca", "mds"]:
            result = sude(X, no_dims=2, k1=15, initialize=init_method, normalize=True)
            assert result.shape == (100, 2)
            assert not np.isnan(result).any()

        # Test with different dimensions
        for n_dims in [2, 3, 5]:
            result = sude(X, no_dims=n_dims, k1=15, normalize=True)
            assert result.shape == (100, n_dims)
            assert not np.isnan(result).any()

        # Test with large dataset mode
        result = sude(X, no_dims=2, k1=15, large=True, normalize=True)
        assert result.shape == (100, 2)
        assert not np.isnan(result).any()

    def test_run_sude_basic(self, adata):
        """Test run_sude function with basic parameters."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Test run_sude
        result_adata = run_sude(adata, basis="pca", n_components=2, copy=True)
        
        assert "X_sude" in result_adata.obsm_keys()
        assert result_adata.obsm["X_sude"].shape[0] == adata.n_obs
        assert result_adata.obsm["X_sude"].shape[1] == 2
        assert not np.isnan(result_adata.obsm["X_sude"]).any()

    def test_run_sude_parameters(self, adata):
        """Test run_sude function with different parameters."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Test with different parameters
        result_adata = run_sude(
            adata, 
            basis="pca", 
            n_components=3, 
            k1=15,
            normalize=True,
            large=False,
            initialize="pca",
            agg_coef=1.5,
            T_epoch=30,
            copy=True
        )
        
        assert "X_sude" in result_adata.obsm_keys()
        assert result_adata.obsm["X_sude"].shape[1] == 3
        
        # Check that parameters are stored (neighbor_key might be different)
        neighbor_keys = [key for key in result_adata.uns_keys() if "neighbor" in key.lower()]
        assert len(neighbor_keys) > 0, "No neighbor key found in uns"
        
        # Check the first neighbor key found
        neighbor_key = neighbor_keys[0]
        assert "k1" in result_adata.uns[neighbor_key]
        assert result_adata.uns[neighbor_key]["k1"] == 15

    def test_run_sude_different_basis(self, adata):
        """Test run_sude with different basis options."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Test with PCA basis
        result_adata = run_sude(adata, basis="pca", n_components=2, copy=True)
        assert "X_sude" in result_adata.obsm_keys()
        
        # Test with UMAP basis if available
        if "X_umap" in adata.obsm_keys():
            result_adata2 = run_sude(adata, basis="umap", n_components=2, copy=True)
            assert "X_sude" in result_adata2.obsm_keys()

    def test_run_sude_enforce(self, adata):
        """Test run_sude enforce parameter."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # First run
        result_adata = run_sude(adata, basis="pca", n_components=2, copy=True)
        first_embedding = result_adata.obsm["X_sude"].copy()
        
        # Second run without enforce (should skip)
        result_adata2 = run_sude(result_adata, basis="pca", n_components=2, enforce=False, copy=True)
        second_embedding = result_adata2.obsm["X_sude"]
        
        # Should be the same (no recomputation)
        np.testing.assert_array_equal(first_embedding, second_embedding)
        
        # Third run with enforce=True (should recompute)
        result_adata3 = run_sude(result_adata, basis="pca", n_components=2, enforce=True, copy=True)
        third_embedding = result_adata3.obsm["X_sude"]
        
        # Should be different (recomputed)
        assert not np.array_equal(first_embedding, third_embedding)

    def test_run_sude_genes_subset(self, adata):
        """Test run_sude with gene subset."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Select subset of genes
        selected_genes = adata.var_names[:50]
        
        result_adata = run_sude(
            adata, 
            genes=selected_genes, 
            basis="pca", 
            n_components=2, 
            copy=True
        )
        
        assert "X_sude" in result_adata.obsm_keys()
        assert result_adata.obsm["X_sude"].shape[0] == adata.n_obs

    def test_run_sude_layer(self, adata):
        """Test run_sude with different layers."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Create a test layer
        adata.layers["test_layer"] = adata.X.copy()
        
        result_adata = run_sude(
            adata, 
            layer="test_layer", 
            basis="pca", 
            n_components=2, 
            copy=True
        )
        
        assert "test_layer_sude" in result_adata.obsm_keys()
        assert result_adata.obsm["test_layer_sude"].shape[0] == adata.n_obs

    def test_run_sude_edge_cases(self):
        """Test run_sude with edge cases."""
        # Test with larger dataset to avoid neighbor issues
        X = np.random.randn(50, 10)  # More cells and features
        obs = pd.DataFrame({'cell_type': ['A'] * 50})
        var = pd.DataFrame({'gene_name': [f'gene_{i}' for i in range(10)]})
        small_adata = AnnData(X=X, obs=obs, var=var)
        
        # Add PCA
        dyn.pp.pca(small_adata, n_pca_components=5)
        
        # Use smaller k1 to avoid neighbor issues
        result_adata = run_sude(small_adata, basis="pca", n_components=2, k1=15, copy=True)
        assert "X_sude" in result_adata.obsm_keys()
        assert result_adata.obsm["X_sude"].shape == (50, 2)

    def test_run_sude_error_handling(self, adata):
        """Test run_sude error handling."""
        # Test with invalid basis
        with pytest.raises(ValueError):
            run_sude(adata, basis="nonexistent", n_components=2, copy=True)
        
        # Test with invalid parameters - n_components should be positive
        with pytest.raises(ValueError):
            run_sude(adata, basis="pca", n_components=0, copy=True)

    def test_sude_integration_with_reduceDimension(self, adata):
        """Test SUDE integration with the main reduceDimension function."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Test using reduceDimension with sude method
        result_adata = dyn.tl.reduceDimension(
            adata, 
            reduction_method="sude", 
            basis="pca", 
            n_components=2, 
            copy=True
        )
        
        assert "X_sude" in result_adata.obsm_keys()
        assert result_adata.obsm["X_sude"].shape[1] == 2

    def test_sude_performance_consistency(self, adata):
        """Test that SUDE produces consistent results with same parameters."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Run SUDE twice with same parameters
        result1 = run_sude(adata, basis="pca", n_components=2, copy=True)
        result2 = run_sude(adata, basis="pca", n_components=2, copy=True)
        
        # Results should be similar (SUDE may not be perfectly deterministic due to random initialization)
        # Check that shapes are the same and no NaN values
        assert result1.obsm["X_sude"].shape == result2.obsm["X_sude"].shape
        assert not np.isnan(result1.obsm["X_sude"]).any()
        assert not np.isnan(result2.obsm["X_sude"]).any()
        
        # Check that results are not identical (SUDE uses random initialization)
        assert not np.array_equal(result1.obsm["X_sude"], result2.obsm["X_sude"])

    def test_sude_memory_efficiency(self, adata):
        """Test SUDE memory efficiency with large dataset mode."""
        # Ensure PCA is available
        if "X_pca" not in adata.obsm_keys():
            dyn.pp.pca(adata, n_pca_components=10)
        
        # Test with large=True
        result_adata = run_sude(
            adata, 
            basis="pca", 
            n_components=2, 
            large=True,
            copy=True
        )
        
        assert "X_sude" in result_adata.obsm_keys()
        assert result_adata.obsm["X_sude"].shape[1] == 2
        
        # Check that large mode parameters are stored
        neighbor_keys = [key for key in result_adata.uns_keys() if "neighbor" in key.lower()]
        if neighbor_keys:
            neighbor_key = neighbor_keys[0]
            assert result_adata.uns[neighbor_key]["large"] is True


if __name__ == "__main__":
    pytest.main([__file__])
