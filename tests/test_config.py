import dynamo as dyn

from dynamo.dynamo_logger import main_tqdm, main_critical, main_warning, main_info


DEFAULT_MODE = "verbose"


def test_config_change():
    assert dyn.config.data_store_mode == "verbose"
    dyn.config.data_store_mode = "succint"

    # change global values wont change other configs
    assert dyn.config.data_store_mode == "succint"
    assert dyn.config.keep_filtered_cells == True
    assert dyn.config.keep_filtered_genes == True
    assert dyn.config.keep_raw_layers == True

    # update data store mode will update others
    dyn.config.update_data_store_mode("succint")
    assert dyn.config.keep_filtered_cells == False
    assert dyn.config.keep_filtered_genes == False
    assert dyn.config.keep_raw_layers == False

    import dynamo.configuration as imported_config

    assert imported_config.data_store_mode == "succint"


if __name__ == "__main__":
    test_config_change()
