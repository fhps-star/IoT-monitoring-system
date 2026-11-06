#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"

#include "esp_err.h"
#include "esp_rom_sys.h"
#include "esp_random.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "mqtt_client.h"
#include "esp_http_client.h"
#include "esp_crt_bundle.h"

#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include "lwip/netdb.h"

// =====================================================
// CONFIGURATION
// =====================================================

#define WIFI_SSID        "Wokwi-GUEST"
#define WIFI_PASSWORD    ""

// ---- PROTOCOL SELECTION (compare MQTT vs HTTP vs CoAP) ----
#define PROTO_MQTT   0
#define PROTO_HTTP   1
#define PROTO_COAP   2
#define PROTOCOL     PROTO_HTTP   // <-- Change to PROTO_MQTT, PROTO_HTTP, or PROTO_COAP

// MQTT Broker & Topics
#define MQTT_BROKER_URI  "mqtt://broker.hivemq.com:1883"
#define MQTT_TOPIC       "iiot/motor_01/telemetry"
#define MQTT_PING_TOPIC  "iiot/motor_01/ping"

// HTTP Backend URL (Change IP to backend host machine if testing on LAN/Wokwi)
#define HTTP_URL             "http://10.0.2.2:8000/api/telemetry"
#define HTTP_TIMEOUT_MS      3000
#define HTTP_REUSE_CONNECTION 1

// CoAP Server (Change IP to backend host machine, UDP Port 5683)
#define COAP_SERVER_IP       "10.0.2.2"
#define COAP_SERVER_PORT     5683
#define COAP_TIMEOUT_MS      3000

#define I2C_PORT         I2C_NUM_0
#define SDA_PIN          GPIO_NUM_21
#define SCL_PIN          GPIO_NUM_22

#define MPU_ADDR         0x68
#define DS18B20_PIN      GPIO_NUM_4

// Current sensor (potentiometer on GPIO34 = ADC1 channel 6)
#define CURRENT_ADC_CHANNEL  ADC_CHANNEL_6
#define CURRENT_MAX_A        2.0f
#define CURRENT_SAMPLES      16

// Stepper motor (A4988)
#define MOTOR_STEP_PIN       GPIO_NUM_25
#define MOTOR_DIR_PIN        GPIO_NUM_26
#define MOTOR_EN_PIN         GPIO_NUM_27
#define MOTOR_STEPS_PER_REV  200
#define MOTOR_MICROSTEPS     1
#define MOTOR_TARGET_RPM     60.0f

#define MOTOR_AUTO_SPEED       1
#define MOTOR_RPM_MIN          30.0f
#define MOTOR_RPM_MAX          90.0f
#define MOTOR_SPEED_PERIOD_S   60.0f

#define SIMULATE_VARIATION   1
#define HIGH_VIBRATION_MODE  0

#define MONITOR_INTERVAL_MS  1000

// =====================================================
// GLOBAL VARIABLES & EVENT GROUPS
// =====================================================

static EventGroupHandle_t wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static esp_mqtt_client_handle_t mqtt_client = NULL;
static bool mqtt_connected = false;

static esp_http_client_handle_t http_client = NULL;

static adc_oneshot_unit_handle_t adc1_handle;

// Ping RTT measurement (for MQTT ping-pong)
static int64_t ping_start_us = 0;
static float mqtt_last_rtt_ms = -1.0f;

// HTTP last measured RTT
static float http_last_rtt_ms = -1.0f;

// CoAP last measured RTT
static float coap_last_rtt_ms = -1.0f;

// Global Sequence Tracker
static uint32_t telemetry_sequence = 0;

// Reconnection Time Tracker
static int64_t disconnect_start_us = 0;
static float reconnection_time_ms = -1.0f;

// Telemetry structure
typedef struct {
    float temp_c;
    float accel_x;
    float accel_y;
    float accel_z;
    float gyro_x;
    float gyro_y;
    float gyro_z;
    float vib_rms;
    float current_a;
    float target_rpm;
} sensor_data_t;

static sensor_data_t latest_sensors;
static SemaphoreHandle_t sensors_mutex;

// =====================================================
// COAP CLIENT IMPLEMENTATION (RFC 7252 CON Request + ACK)
// =====================================================

static bool coap_post_telemetry(const char *json_payload, float *out_rtt_ms) {
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        printf("[COAP] Unable to create socket: errno %d\n", errno);
        return false;
    }

    struct timeval tv;
    tv.tv_sec = COAP_TIMEOUT_MS / 1000;
    tv.tv_usec = (COAP_TIMEOUT_MS % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(COAP_SERVER_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(COAP_SERVER_PORT);

    // Build RFC 7252 CoAP CON POST packet
    // Header: Ver=1, Type=CON(0), TKL=0 -> 0x40
    // Code = 0.02 POST -> 0x02
    uint16_t msg_id = (uint16_t)(telemetry_sequence & 0xFFFF);
    uint8_t packet[1024];
    int idx = 0;

    packet[idx++] = 0x40;                      // Ver 1, Type CON, TKL 0
    packet[idx++] = 0x02;                      // Code 0.02 (POST)
    packet[idx++] = (msg_id >> 8) & 0xFF;      // Msg ID MSB
    packet[idx++] = msg_id & 0xFF;             // Msg ID LSB

    // Uri-Path Option (Option 11): Delta=11, Length=9, Value="telemetry"
    packet[idx++] = (11 << 4) | 9;
    memcpy(&packet[idx], "telemetry", 9);
    idx += 9;

    // Content-Format Option (Option 12): Delta=1, Length=1, Value=50 (application/json)
    packet[idx++] = (1 << 4) | 1;
    packet[idx++] = 50;

    // Payload Marker: 0xFF
    packet[idx++] = 0xFF;

    // Payload body
    int payload_len = strlen(json_payload);
    if (idx + payload_len >= sizeof(packet)) {
        payload_len = sizeof(packet) - idx - 1;
    }
    memcpy(&packet[idx], json_payload, payload_len);
    idx += payload_len;

    int64_t t0 = esp_timer_get_time();
    int err = sendto(sock, packet, idx, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
    if (err < 0) {
        printf("[COAP] Error during sendto: errno %d\n", errno);
        close(sock);
        return false;
    }

    // Wait for ACK from backend
    uint8_t rx_buffer[128];
    struct sockaddr_in source_addr;
    socklen_t socklen = sizeof(source_addr);
    int len = recvfrom(sock, rx_buffer, sizeof(rx_buffer), 0, (struct sockaddr *)&source_addr, &socklen);

    int64_t t1 = esp_timer_get_time();
    close(sock);

    if (len < 4) {
        printf("[COAP] Timeout or invalid ACK received\n");
        return false;
    }

    // Verify ACK type (Type = 2) and Message ID match
    uint8_t ack_type = (rx_buffer[0] >> 4) & 0x03;
    uint16_t ack_msg_id = (rx_buffer[2] << 8) | rx_buffer[3];

    if (ack_type == 2 && ack_msg_id == msg_id) {
        float rtt = (t1 - t0) / 1000.0f;
        if (out_rtt_ms) *out_rtt_ms = rtt;
        printf("[COAP] ACK received! RTT = %.1f ms\n", rtt);
        return true;
    }

    printf("[COAP] Mismatched ACK packet (type=%d, id=%d)\n", ack_type, ack_msg_id);
    return false;
}

// =====================================================
// HTTP POST IMPLEMENTATION
// =====================================================

static bool http_post_telemetry(const char *json_payload, float *out_rtt_ms) {
    esp_http_client_config_t config = {
        .url = HTTP_URL,
        .method = HTTP_METHOD_POST,
        .timeout_ms = HTTP_TIMEOUT_MS,
        .crt_bundle_attach = esp_crt_bundle_attach,
        .skip_cert_common_name_check = true,
    };

    if (http_client == NULL) {
        http_client = esp_http_client_init(&config);
    }

    esp_http_client_set_url(http_client, HTTP_URL);
    esp_http_client_set_method(http_client, HTTP_METHOD_POST);
    esp_http_client_set_header(http_client, "Content-Type", "application/json");
    esp_http_client_set_header(http_client, "ngrok-skip-browser-warning", "true");
    esp_http_client_set_post_field(http_client, json_payload, strlen(json_payload));

    int64_t t0 = esp_timer_get_time();
    esp_err_t err = esp_http_client_perform(http_client);
    int64_t t1 = esp_timer_get_time();

    if (err == ESP_OK) {
        float rtt = (t1 - t0) / 1000.0f;
        int status_code = esp_http_client_get_status_code(http_client);
        if (out_rtt_ms) *out_rtt_ms = rtt;
        printf("[HTTP] POST successful (status %d), RTT = %.1f ms\n", status_code, rtt);
#if !HTTP_REUSE_CONNECTION
        esp_http_client_cleanup(http_client);
        http_client = NULL;
#endif
        return true;
    } else {
        printf("[HTTP] POST failed: %s\n", esp_err_to_name(err));
        esp_http_client_cleanup(http_client);
        http_client = NULL;
        return false;
    }
}

// =====================================================
// TELEMETRY PUBLISH DISPATCHER
// =====================================================

static void telemetry_publish(const sensor_data_t *s) {
    telemetry_sequence++;
    char json[512];
    uint32_t free_heap = (uint32_t)esp_get_free_heap_size();
    int64_t now_ms = esp_timer_get_time() / 1000;

    const char *proto_name = "MQTT";
    float current_rtt = -1.0f;

#if PROTOCOL == PROTO_HTTP
    proto_name = "HTTP";
    current_rtt = http_last_rtt_ms;
#elif PROTOCOL == PROTO_COAP
    proto_name = "CoAP";
    current_rtt = coap_last_rtt_ms;
#else
    proto_name = "MQTT";
    current_rtt = mqtt_last_rtt_ms;
#endif

    // Format telemetry JSON with all required metrics
    if (current_rtt >= 0.0f) {
        snprintf(json, sizeof(json),
            "{\"device_id\":\"motor_01\",\"motor_id\":\"motor_01\","
            "\"sequence\":%lu,\"protocol\":\"%s\","
            "\"timestamp_ms\":%lld,\"rtt_ms\":%.2f,\"free_heap\":%lu,"
            "\"temperature\":%.2f,\"vibration\":%.3f,\"current\":%.3f,"
            "\"rpm\":%.0f,\"accel_x\":%.3f,\"accel_y\":%.3f,\"accel_z\":%.3f}",
            (unsigned long)telemetry_sequence, proto_name,
            now_ms, current_rtt, (unsigned long)free_heap,
            s->temp_c, s->vib_rms, s->current_a,
            s->target_rpm, s->accel_x, s->accel_y, s->accel_z);
    } else {
        snprintf(json, sizeof(json),
            "{\"device_id\":\"motor_01\",\"motor_id\":\"motor_01\","
            "\"sequence\":%lu,\"protocol\":\"%s\","
            "\"timestamp_ms\":%lld,\"free_heap\":%lu,"
            "\"temperature\":%.2f,\"vibration\":%.3f,\"current\":%.3f,"
            "\"rpm\":%.0f,\"accel_x\":%.3f,\"accel_y\":%.3f,\"accel_z\":%.3f}",
            (unsigned long)telemetry_sequence, proto_name,
            now_ms, (unsigned long)free_heap,
            s->temp_c, s->vib_rms, s->current_a,
            s->target_rpm, s->accel_x, s->accel_y, s->accel_z);
    }

#if PROTOCOL == PROTO_MQTT
    if (mqtt_connected) {
        esp_mqtt_client_publish(mqtt_client, MQTT_TOPIC, json, 0, 0, 0);
        printf("[MQTT] Published seq=%lu to %s\n", (unsigned long)telemetry_sequence, MQTT_TOPIC);
    }
#elif PROTOCOL == PROTO_HTTP
    float rtt = 0.0f;
    if (http_post_telemetry(json, &rtt)) {
        http_last_rtt_ms = rtt;
    }
#elif PROTOCOL == PROTO_COAP
    float rtt = 0.0f;
    if (coap_post_telemetry(json, &rtt)) {
        coap_last_rtt_ms = rtt;
    }
#endif
}

// =====================================================
// WIFI & MQTT HANDLERS
// =====================================================

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        disconnect_start_us = esp_timer_get_time();
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        printf("[WIFI] Disconnected, reconnecting...\n");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        if (disconnect_start_us > 0) {
            reconnection_time_ms = (esp_timer_get_time() - disconnect_start_us) / 1000.0f;
            printf("[WIFI] Reconnected in %.1f ms\n", reconnection_time_ms);
        }
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
        printf("[WIFI] Connected with IP\n");
    }
}

static void wifi_init_sta(void) {
    wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_mqtt_event_handle_t event = event_data;
    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            mqtt_connected = true;
            printf("[MQTT] Connected to broker\n");
            esp_mqtt_client_subscribe(event->client, MQTT_PING_TOPIC, 0);
            break;
        case MQTT_EVENT_DISCONNECTED:
            mqtt_connected = false;
            printf("[MQTT] Disconnected from broker\n");
            break;
        case MQTT_EVENT_DATA:
            if (strncmp(event->topic, MQTT_PING_TOPIC, event->topic_len) == 0) {
                if (ping_start_us > 0) {
                    mqtt_last_rtt_ms = (esp_timer_get_time() - ping_start_us) / 1000.0f;
                    printf("[MQTT] Ping RTT = %.1f ms\n", mqtt_last_rtt_ms);
                    ping_start_us = 0;
                }
            }
            break;
        default:
            break;
    }
}

static void mqtt_app_start(void) {
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URI,
    };
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(mqtt_client);
}

// =====================================================
// SENSORS & MOTOR TASKS
// =====================================================

static void monitor_task(void *pvParameters) {
    while (1) {
        xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, false, true, portMAX_DELAY);

        // Update simulated/sensor telemetry
        if (xSemaphoreTake(sensors_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            latest_sensors.temp_c = 38.0f + ((float)(esp_random() % 200) / 100.0f);
            latest_sensors.vib_rms = 0.25f + ((float)(esp_random() % 100) / 1000.0f);
            latest_sensors.current_a = 0.50f + ((float)(esp_random() % 100) / 1000.0f);
            latest_sensors.target_rpm = MOTOR_TARGET_RPM;
            latest_sensors.accel_x = 0.05f;
            latest_sensors.accel_y = -0.02f;
            latest_sensors.accel_z = 0.98f;

            telemetry_publish(&latest_sensors);
            xSemaphoreGive(sensors_mutex);
        }

#if PROTOCOL == PROTO_MQTT
        if (mqtt_connected && ping_start_us == 0) {
            ping_start_us = esp_timer_get_time();
            char ping_msg[32];
            snprintf(ping_msg, sizeof(ping_msg), "%lld", ping_start_us);
            esp_mqtt_client_publish(mqtt_client, MQTT_PING_TOPIC, ping_msg, 0, 0, 0);
        }
#endif

        vTaskDelay(pdMS_TO_TICKS(MONITOR_INTERVAL_MS));
    }
}

// =====================================================
// MAIN ENTRY POINT
// =====================================================

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    sensors_mutex = xSemaphoreCreateMutex();
    wifi_init_sta();

#if PROTOCOL == PROTO_MQTT
    mqtt_app_start();
#endif

    xTaskCreatePinnedToCore(monitor_task, "monitor_task", 4096, NULL, 5, NULL, 1);
    printf("[MAIN] ESP32 Protocol Benchmark Station initialized! Protocol=%s\n",
#if PROTOCOL == PROTO_HTTP
        "HTTP"
#elif PROTOCOL == PROTO_COAP
        "CoAP"
#else
        "MQTT"
#endif
    );
}

