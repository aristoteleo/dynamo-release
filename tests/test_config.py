import dynamo as dyn

from dynamo.dynamo_logger import main_tqdm, main_critical, main_warning, main_info


DEFAULT_MODE = "verbose"


def test_config_change():
    assert dyn.config.DynamoSaveConfig.data_store_mode == "full"
    dyn.config.DynamoSaveConfig.data_store_mode = "succint"

    # change class static values wont change config variables
    assert dyn.config.DynamoSaveConfig.data_store_mode == "succint"
    assert dyn.config.DynamoSaveConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoSaveConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoSaveConfig.recipe_keep_raw_layers_default == False

    # update data store mode will update config variables as well

    # test succint mode
    dyn.config.DynamoSaveConfig.update_data_store_mode("succint")
    assert dyn.config.DynamoSaveConfig.recipe_keep_filtered_cells_default == False
    assert dyn.config.DynamoSaveConfig.recipe_keep_filtered_genes_default == False
    assert dyn.config.DynamoSaveConfig.recipe_keep_raw_layers_default == False
    assert dyn.config.DynamoSaveConfig.recipe_monocle_keep_filtered_cells_default == False
    assert dyn.config.DynamoSaveConfig.recipe_monocle_keep_filtered_genes_default == False

    # test full mode
    dyn.config.DynamoSaveConfig.update_data_store_mode("full")
    assert dyn.config.DynamoSaveConfig.recipe_keep_filtered_cells_default == False
    assert dyn.config.DynamoSaveConfig.recipe_keep_filtered_genes_default == False
    assert dyn.config.DynamoSaveConfig.recipe_keep_raw_layers_default == False
    assert dyn.config.DynamoSaveConfig.recipe_monocle_keep_filtered_cells_default == True
    assert dyn.config.DynamoSaveConfig.recipe_monocle_keep_filtered_cells_default == True

    dyn.config.DynamoSaveConfig.update_data_store_mode("succint")
    import dynamo.configuration as imported_config

    assert imported_config.DynamoSaveConfig.data_store_mode == "succint"


if __name__ == "__main__":
    test_config_change()
