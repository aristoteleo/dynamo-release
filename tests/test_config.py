import dynamo as dyn

from dynamo.dynamo_logger import main_tqdm, main_critical, main_warning, main_info


DEFAULT_MODE = "verbose"


def test_config_change():
    assert dyn.config.DynamoConfig.data_store_mode == "verbose"
    dyn.config.DynamoConfig.data_store_mode = "succint"

    # change global values wont change other configs
    assert dyn.config.DynamoConfig.data_store_mode == "succint"
    assert dyn.config.DynamoConfig.keep_filtered_cells_default == True
    assert dyn.config.DynamoConfig.keep_filtered_genes_default == True
    assert dyn.config.DynamoConfig.keep_raw_layers_default == True

    # update data store mode will update others
    dyn.config.DynamoConfig.update_data_store_mode("succint")
    assert dyn.config.DynamoConfig.keep_filtered_cells_default == False
    assert dyn.config.DynamoConfig.keep_filtered_genes_default == False
    assert dyn.config.DynamoConfig.keep_raw_layers_default == False

    import dynamo.configuration as imported_config

    assert imported_config.DynamoConfig.data_store_mode == "succint"


if __name__ == "__main__":
    test_config_change()
