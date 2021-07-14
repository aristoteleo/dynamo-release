import dynamo as dyn

from dynamo.dynamo_logger import main_tqdm, main_critical, main_warning, main_info


DEFAULT_MODE = "verbose"


def test_config_change():
    assert dyn.config.DynamoDataConfig.data_store_mode == "full"
    dyn.config.DynamoDataConfig.data_store_mode = "succint"

    # change class static values wont change config variables
    assert dyn.config.DynamoDataConfig.data_store_mode == "succint"
    assert dyn.config.DynamoDataConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoDataConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoDataConfig.recipe_keep_raw_layers_default == False

    # update data store mode will update config variables as well

    # test succint mode
    dyn.config.DynamoDataConfig.update_data_store_mode("succint")
    assert dyn.config.DynamoDataConfig.recipe_keep_filtered_cells_default == False
    assert dyn.config.DynamoDataConfig.recipe_keep_filtered_genes_default == False
    assert dyn.config.DynamoDataConfig.recipe_keep_raw_layers_default == False
    assert dyn.config.DynamoDataConfig.recipe_monocle_keep_filtered_cells_default == False
    assert dyn.config.DynamoDataConfig.recipe_monocle_keep_filtered_genes_default == False

    # test full mode
    dyn.config.DynamoDataConfig.update_data_store_mode("full")
    assert dyn.config.DynamoDataConfig.recipe_keep_filtered_cells_default == False
    assert dyn.config.DynamoDataConfig.recipe_keep_filtered_genes_default == False
    assert dyn.config.DynamoDataConfig.recipe_keep_raw_layers_default == False
    assert dyn.config.DynamoDataConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoDataConfig.recipe_monocle_keep_filtered_cells_default == True

    dyn.config.DynamoDataConfig.update_data_store_mode("succint")
    import dynamo.configuration as imported_config

    assert imported_config.DynamoDataConfig.data_store_mode == "succint"


if __name__ == "__main__":
    test_config_change()
