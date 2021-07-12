import dynamo as dyn

from dynamo.dynamo_logger import main_tqdm, main_critical, main_warning, main_info


DEFAULT_MODE = "verbose"


def test_config_change():
    assert dyn.config.DynamoConfig.data_store_mode == "default"
    dyn.config.DynamoConfig.data_store_mode = "succint"

    # change global values wont change other configs
    assert dyn.config.DynamoConfig.data_store_mode == "succint"
    assert dyn.config.DynamoConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoConfig.keep_raw_layers_default == False

    # update data store mode will update others
    dyn.config.DynamoConfig.update_data_store_mode("succint")
    assert dyn.config.DynamoConfig.keep_filtered_cells_default == False
    assert dyn.config.DynamoConfig.keep_filtered_genes_default == False
    assert dyn.config.DynamoConfig.keep_raw_layers_default == False
    dyn.config.DynamoConfig.update_data_store_mode("full")
    assert dyn.config.DynamoConfig.keep_filtered_cells_default == True
    assert dyn.config.DynamoConfig.keep_filtered_genes_default == True
    assert dyn.config.DynamoConfig.keep_raw_layers_default == True

    import dynamo.configuration as imported_config

    assert imported_config.DynamoConfig.data_store_mode == "full"


if __name__ == "__main__":
    test_config_change()
