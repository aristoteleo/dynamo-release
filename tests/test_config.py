import dynamo as dyn
from dynamo.dynamo_logger import main_critical, main_info, main_tqdm, main_warning

DEFAULT_MODE = "verbose"


def test_config_change():
    assert dyn.config.DynamoAdataConfig.data_store_mode == "full"
    dyn.config.DynamoAdataConfig.data_store_mode = "succinct"

    # change class static values wont change config variables
    assert dyn.config.DynamoAdataConfig.data_store_mode == "succinct"
    assert dyn.config.DynamoAdataConfig.recipe_monocle_keep_filtered_cells == True
    assert dyn.config.DynamoAdataConfig.recipe_monocle_keep_filtered_cells == True
    assert dyn.config.DynamoAdataConfig.recipe_keep_raw_layers == False

    # update data store mode will update config variables as well

    # test succint mode
    dyn.config.DynamoAdataConfig.update_data_store_mode("succinct")
    assert dyn.config.DynamoAdataConfig.recipe_keep_filtered_cells == False
    assert dyn.config.DynamoAdataConfig.recipe_keep_filtered_genes == False
    assert dyn.config.DynamoAdataConfig.recipe_keep_raw_layers == False
    assert dyn.config.DynamoAdataConfig.recipe_monocle_keep_filtered_cells == False
    assert dyn.config.DynamoAdataConfig.recipe_monocle_keep_filtered_genes == False

    # test full mode
    dyn.config.DynamoAdataConfig.update_data_store_mode("full")
    assert dyn.config.DynamoAdataConfig.recipe_keep_filtered_cells == False
    assert dyn.config.DynamoAdataConfig.recipe_keep_filtered_genes == False
    assert dyn.config.DynamoAdataConfig.recipe_keep_raw_layers == False
    assert dyn.config.DynamoAdataConfig.recipe_monocle_keep_filtered_cells == True
    assert dyn.config.DynamoAdataConfig.recipe_monocle_keep_filtered_cells == True

    dyn.config.DynamoAdataConfig.update_data_store_mode("succinct")
    import dynamo.configuration as imported_config

    assert imported_config.DynamoAdataConfig.data_store_mode == "succinct"


if __name__ == "__main__":
    test_config_change()
