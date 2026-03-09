#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <time.h>
#include <sys/time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ── User config ──────────────────────────────────────────────────────────────
#define WIFI_SSID     "215"
#define WIFI_PASSWORD "1234567890"
#define LAPTOP_IP     "192.168.137.239"
#define LAPTOP_PORT   5000
#define CMD_PORT      5001   // laptop → ESP32 control channel
#define OTA_PASSWORD  "esp32ota"
// ─────────────────────────────────────────────────────────────────────────────

// AI-Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// UDP fragmentation
// Packet layout: [4B frame_id][2B chunk_idx][2B total_chunks][8B timestamp_us][payload]
#define CHUNK_SIZE  1400
#define HDR_SIZE      16

struct FrameEnvelope {
    camera_fb_t *fb;
    int64_t      ts_us;
};

static QueueHandle_t frameQueue;
static WiFiUDP       udp;
static WiFiUDP       cmdUdp;
static uint32_t      frameId = 0;

// Live-tunable settings (read by captureTask, written by cmdTask)
static volatile int  g_jpeg_quality = 12;   // 0=best … 63=worst
static volatile int  g_target_fps   = 30;

// ── Camera init ───────────────────────────────────────────────────────────────
static bool initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = g_jpeg_quality;
        config.fb_count     = 2;
    } else {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = g_jpeg_quality;
        config.fb_count     = 1;
        Serial.println("WARN: No PSRAM");
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init FAILED");
        return false;
    }

    sensor_t *s = esp_camera_sensor_get();
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);

    Serial.println("Camera OK");
    return true;
}

// ── NTP sync ──────────────────────────────────────────────────────────────────
static void syncTime() {
    configTime(0, 0, LAPTOP_IP);
    Serial.print("NTP sync");
    struct tm ti;
    int retries = 0;
    while (!getLocalTime(&ti) && retries++ < 20) {
        Serial.print('.');
        delay(500);
    }
    if (retries >= 20) {
        Serial.println("\nWARN: NTP timed out");
    } else {
        char buf[32];
        strftime(buf, sizeof(buf), "%H:%M:%S", &ti);
        Serial.printf(" OK: %s UTC\n", buf);
    }
}

// ── Task: capture at target FPS ───────────────────────────────────────────────
void captureTask(void *) {
    TickType_t lastWake = xTaskGetTickCount();

    while (true) {
        int fps = g_target_fps;
        TickType_t period = pdMS_TO_TICKS(1000 / fps);

        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelayUntil(&lastWake, period);
            continue;
        }

        // Apply any quality change requested via cmd channel
        sensor_t *s = esp_camera_sensor_get();
        s->set_quality(s, g_jpeg_quality);

        struct timeval tv;
        gettimeofday(&tv, NULL);
        FrameEnvelope env = {
            .fb    = fb,
            .ts_us = (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec
        };

        if (xQueueSend(frameQueue, &env, 0) != pdTRUE) {
            esp_camera_fb_return(fb);
        }

        vTaskDelayUntil(&lastWake, period);
    }
}

// ── Task: fragment and send each frame over UDP ───────────────────────────────
void sendTask(void *) {
    static uint8_t pkt[HDR_SIZE + CHUNK_SIZE];

    while (true) {
        FrameEnvelope env;
        if (xQueueReceive(frameQueue, &env, pdMS_TO_TICKS(500)) != pdTRUE) {
            continue;
        }

        const uint8_t *data    = env.fb->buf;
        const size_t   total   = env.fb->len;
        const uint16_t nchunks = (total + CHUNK_SIZE - 1) / CHUNK_SIZE;
        const uint32_t fid     = frameId++;

        for (uint16_t i = 0; i < nchunks; i++) {
            size_t   offset  = (size_t)i * CHUNK_SIZE;
            uint16_t pld_len = (uint16_t)((offset + CHUNK_SIZE <= total)
                                           ? CHUNK_SIZE
                                           : total - offset);

            memcpy(pkt,      &fid,       4);
            memcpy(pkt + 4,  &i,         2);
            memcpy(pkt + 6,  &nchunks,   2);
            memcpy(pkt + 8,  &env.ts_us, 8);
            memcpy(pkt + HDR_SIZE, data + offset, pld_len);

            udp.beginPacket(LAPTOP_IP, LAPTOP_PORT);
            udp.write(pkt, HDR_SIZE + pld_len);
            udp.endPacket();
        }

        esp_camera_fb_return(env.fb);
    }
}

// ── Task: receive control commands from laptop ────────────────────────────────
// Commands (plain text, UDP to port CMD_PORT):
//   q:<0-63>   set JPEG quality   e.g. "q:20"
//   f:<fps>    set target FPS     e.g. "f:15"
void cmdTask(void *) {
    cmdUdp.begin(CMD_PORT);
    Serial.printf("CMD listener on UDP :%d\n", CMD_PORT);

    while (true) {
        int len = cmdUdp.parsePacket();
        if (len <= 0) {
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }

        char buf[32] = {};
        cmdUdp.read(buf, sizeof(buf) - 1);

        if (buf[0] == 'q' && buf[1] == ':') {
            int val = atoi(buf + 2);
            if (val >= 0 && val <= 63) {
                g_jpeg_quality = val;
                Serial.printf("Quality → %d\n", val);
            }
        } else if (buf[0] == 'f' && buf[1] == ':') {
            int val = atoi(buf + 2);
            if (val >= 1 && val <= 60) {
                g_target_fps = val;
                Serial.printf("FPS → %d\n", val);
            }
        } else {
            Serial.printf("Unknown cmd: %s\n", buf);
        }
    }
}

// ── Task: handle OTA updates ──────────────────────────────────────────────────
void otaTask(void *) {
    ArduinoOTA.setPassword(OTA_PASSWORD);

    ArduinoOTA.onStart([]() {
        Serial.println("OTA start — suspending stream");
    });
    ArduinoOTA.onEnd([]() {
        Serial.println("OTA done — rebooting");
    });
    ArduinoOTA.onError([](ota_error_t e) {
        Serial.printf("OTA error [%u]\n", e);
    });

    ArduinoOTA.begin();
    Serial.println("OTA ready on port 3232");

    while (true) {
        ArduinoOTA.handle();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ── Arduino entry points ──────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n== ESP32-CAM UDP stream ==");

    if (!initCamera()) {
        Serial.println("FATAL: halting");
        while (true) delay(1000);
    }

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("Connecting to %s", WIFI_SSID);
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print('.');
        delay(500);
    }
    Serial.printf("\nWi-Fi OK — IP: %s\n", WiFi.localIP().toString().c_str());

    syncTime();

    udp.begin(LAPTOP_PORT);

    frameQueue = xQueueCreate(2, sizeof(FrameEnvelope));
    xTaskCreatePinnedToCore(captureTask, "capture", 4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(sendTask,    "send",    8192, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(cmdTask,     "cmd",     4096, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(otaTask,     "ota",     8192, NULL, 1, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
