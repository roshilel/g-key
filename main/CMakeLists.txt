idf_component_register(SRCS "ble_hidd_demo_main.c"
                            "esp_hidd_prf_api.c"
                            "hid_dev.c"
                            "hid_device_le_prf.c"
                            "model/nextkeymodel.cc"
                            "model_runner.cc"
                    PRIV_REQUIRES bt nvs_flash esp_driver_gpio esp-tflite-micro
                    INCLUDE_DIRS ".")

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-unused-const-variable)
