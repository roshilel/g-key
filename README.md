# AI-Powered Bluetooth Keyboard Predictor

This project turns an ESP32 into a wireless BLE HID keyboard predictor with a built-in TensorFlow Lite next-character prediction model. As you type via UART, the 1D Convolutional model predicts your next keystroke and can type it out to your connected Bluetooth device.

## Environment Setup

This project requires the esp-idf. Once installed, copy this repository into the your esp-idf directory.

## Automatic Hardware Setup

All hardware configuration settings different from the defaults are in `sdkconfig.defaults` and will be automatically applied when the project is built

## Manual Hardware Setup

If the automatic setup is not working for some reason, here is how you copy the setup manually. All the following changes are made in the configuration, which can be opened by running

```bash
idf.py menuconfig
```

1. Enable PSRAM

Navigate to "Component config" -> "ESP PSRAM" and enable "Support for external, SPI-connected RAM"

2. Use custom flash memory partition.

Navigate to "Partition Table" -> "Partition Table(...)" and change it to "Custom partition table CSV"

3. Enable 4MB Flash

Navigate to "Serial flasher config" -> "Flash size(...)" and select "4 MB"

### Build and Flash

To build the project run `idf.py build`

Run `idf.py -p PORT flash monitor` to build, flash and monitor the project.

(To exit the serial monitor, type ``Ctrl-]``.)

## Example Output

```
I (29) boot: ESP-IDF v6.1-dev-1406-g1e87d43f1a 2nd stage bootloader
I (29) boot: compile time Feb 25 2026 20:39:02
I (29) boot: Multicore bootloader
I (32) boot: chip revision: v3.1
I (35) boot.esp32: SPI Speed      : 40MHz
I (39) boot.esp32: SPI Mode       : DIO
I (42) boot.esp32: SPI Flash Size : 4MB
I (46) boot: Enabling RNG early entropy source...
I (50) boot: Partition Table:
I (53) boot: ## Label            Usage          Type ST Offset   Length
I (59) boot:  0 nvs              WiFi data        01 02 00009000 00006000
I (66) boot:  1 phy_init         RF data          01 01 0000f000 00001000
I (72) boot:  2 factory          factory app      00 00 00010000 00300000
I (79) boot: End of partition table
I (82) esp_image: segment 0: paddr=00010020 vaddr=3f400020 size=d4738h (870200) map
I (400) esp_image: segment 1: paddr=000e4760 vaddr=3ffbdb60 size=05480h ( 21632) load
I (408) esp_image: segment 2: paddr=000e9be8 vaddr=40080000 size=06430h ( 25648) load
I (419) esp_image: segment 3: paddr=000f0020 vaddr=400d0020 size=82198h (532888) map
I (609) esp_image: segment 4: paddr=001721c0 vaddr=40086430 size=12c4ch ( 76876) load
I (641) esp_image: segment 5: paddr=00184e14 vaddr=50000000 size=00028h (    40) load
I (653) boot: Loaded app from partition at offset 0x10000
I (653) boot: Disabling RNG early entropy source...
I (664) quad_psram: This chip is ESP32-D0WD
I (666) esp_psram: Found 8MB PSRAM device
I (666) esp_psram: Speed: 40MHz
I (666) esp_psram: PSRAM initialized, cache is in low/high (2-core) mode.
W (671) esp_psram: Virtual address not enough for PSRAM, map as much as we can. 4MB is mapped
I (679) cpu_start: Multicore app
I (1589) esp_psram: SPI SRAM memory test OK
I (1597) cpu_start: GPIO 3 and 1 are used as console UART I/O pins
I (1597) cpu_start: Pro cpu start user code
I (1597) cpu_start: cpu freq: 160000000 Hz
I (1599) app_init: Application information:
I (1603) app_init: Project name:     hidd_demos
I (1608) app_init: App version:      97119e8-dirty
I (1612) app_init: Compile time:     Feb 27 2026 20:54:32
I (1617) app_init: ELF file SHA256:  85e9baa84...
I (1622) app_init: ESP-IDF:          v6.1-dev-1406-g1e87d43f1a
I (1627) efuse_init: Min chip rev:     v0.0
I (1631) efuse_init: Max chip rev:     v3.99 
I (1635) efuse_init: Chip rev:         v3.1
I (1640) heap_init: Initializing. RAM available for dynamic allocation:
I (1646) heap_init: At 3FFAFF10 len 000000F0 (0 KiB): DRAM
I (1651) heap_init: At 3FFB6388 len 00001C78 (7 KiB): DRAM
I (1656) heap_init: At 3FFB9A20 len 00004108 (16 KiB): DRAM
I (1661) heap_init: At 3FFC7DD8 len 00018228 (96 KiB): DRAM
I (1666) heap_init: At 3FFE0440 len 00003AE0 (14 KiB): D/IRAM
I (1672) heap_init: At 3FFE4350 len 0001BCB0 (111 KiB): D/IRAM
I (1678) heap_init: At 4009907C len 00006F84 (27 KiB): IRAM
I (1683) esp_psram: Adding pool of 4096K of PSRAM memory to heap allocator
I (1691) spi_flash: detected chip: generic
I (1693) spi_flash: flash io: dio
I (1697) coexist: coex firmware version: 7260f71
I (1702) main_task: Started on CPU0
I (1712)I (1712) esp32_keybooard: Initialized UART...
Attempting to allocate 61440 bytes in PSRAM...
Invoked successfully
I (2352) BTDM_INIT: BT controller compile version [c28b205]
I (2352) BTDM_INIT: Using main XTAL as clock source
I (2352) BTDM_INIT: Bluetooth MAC: ec:c9:ff:cd:58:2e
I (2362) phy_init: phy_version 4863,a3a4459,Oct 28 2025,14:30:06
W (2762) BT_BTM: BTM_BleWriteAdvData, Partial data write into ADV
I (2772) HID_LE_PRF: esp_hidd_prf_cb_hdl(), start added the hid service to the stack database. incl_handle = 40
I (2782) HID_LE_PRF: hid svc handle = 2d
I (2792) main_task: Returned from app_main()
I (3012) HID_LE_PRF: HID connection establish, conn_id = 0
I (3012) esp32_keybooard: ESP_HIDD_EVENT_BLE_CONNECT
I (3022) esp32_keybooard: remote BD_ADDR: bcf4d4b02e94
I (3022) esp32_keybooard: address type = 0
I (3022) esp32_keybooard: pair status = success
I (3032) esp32_keybooard: secure connection established.
I (3062) esp32_keybooard: ESP_HIDD_EVENT_BLE_LED_REPORT_WRITE_EVT
I (3062) esp32_keybooard: 04
I (10452) esp32_keybooard: input character: t
'h': 58
Confidence: 0.649456
I (11312) esp32_keybooard: predicted character: h
```

## Terminal Usage

1. Connect to the ESP32 via bluetooth on the device you ran `idf.py flash monitor` on. It should appear as "ESP-32 KEYBOARD"

2. Type any valid character into the terminal and it will send it via UART to the ESP32 and automatically prompt model inference(takes ~1 sec).

*As shown in the above log, the terminal will immediately output the character you typed followed by the predicted character('h'), its token id(58), as well as the models confidence in the prediction(0.649456).

2. When you want the AI to type for you, press an **invalid/unsupported character** (like `~` or the `Tab` key) in your terminal.
