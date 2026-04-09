#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <time.h>
#include <sys/time.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// ── User config ──────────────────────────────────────────────────────────────
#define WIFI_SSID     "iitm_wifi_"
#define WIFI_PASSWORD "12345678"
#define OTA_PASSWORD  "esp32ota"

#ifndef CAM_ID
  #define CAM_ID 1
#endif

#define _STR(x) #x
#define STR(x) _STR(x)

#define LAPTOP_PORT      (5000 + (CAM_ID - 1) * 2)  // cam1=5000  cam2=5002
#define CMD_PORT         (5001 + (CAM_ID - 1) * 2)  // cam1=5001  cam2=5003
#define DISCOVERY_PORT   5004                        // shared beacon channel
#define TIMESYNC_PORT    5005                        // NTP-style round-trip time sync
#define LOG_PORT         5010                        // UDP debug log to laptop
#define OTA_HOSTNAME     "esp32cam-" STR(CAM_ID)
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
static WiFiUDP       discUdp;
static WiFiUDP       logUdp;
static WiFiUDP       tsUdp;
static uint32_t      frameId = 0;

// Laptop IP discovered at runtime via beacon — no hardcoded IP needed
static char          g_laptop_ip[24]      = "";  // empty until beacon received
static char          g_last_laptop_ip[24] = "";  // last known — kept for logging after drop

// Live-tunable settings (read by captureTask, written by cmdTask)
// Defaults match cam_settings.json (pushed by receiver on connect)
static volatile int  g_jpeg_quality = 6;
static volatile int  g_target_fps   = 14;
static volatile int  g_framesize    = FRAMESIZE_HD;    // 1280×720
// Set during OTA — pauses capture+send so WiFi bandwidth goes to the upload
static volatile bool g_ota_active   = false;

// ── UDP log helper ────────────────────────────────────────────────────────────
// Prints to Serial AND sends a UDP packet to the laptop log listener (port 5010)
static void udp_log(const char *fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    Serial.println(buf);

    // Use current IP if known, otherwise fall back to last known IP
    const char *log_ip = g_laptop_ip[0] ? g_laptop_ip : g_last_laptop_ip;
    if (log_ip[0]) {
        char tagged[280];
        snprintf(tagged, sizeof(tagged), "CAM%d|%s", CAM_ID, buf);
        logUdp.beginPacket(log_ip, LOG_PORT);
        logUdp.write((const uint8_t *)tagged, strlen(tagged));
        logUdp.endPacket();
    }
}

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
        // Allocate buffer for the largest possible frame so set_framesize()
        // can switch to any resolution at runtime without buffer overflow.
        config.frame_size   = FRAMESIZE_UXGA;
        config.jpeg_quality = g_jpeg_quality;
        config.fb_count     = 2;
    } else {
        // No PSRAM — cap at SVGA to fit in internal RAM
        config.frame_size   = FRAMESIZE_SVGA;
        config.jpeg_quality = g_jpeg_quality;
        config.fb_count     = 1;
        Serial.println("WARN: No PSRAM — max res SVGA");
    }

    if (esp_camera_init(&config) != ESP_OK) {
        Serial.println("Camera init FAILED");
        return false;
    }

    sensor_t *s = esp_camera_sensor_get();
    // ── Sensor defaults (match cam_settings.json) ─────────────────────────────
    s->set_framesize(s, (framesize_t)g_framesize);  // HD 1280×720
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);
    s->set_ae_level(s, 0);
    s->set_whitebal(s, 1);       // AWB on
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);        // auto WB
    s->set_exposure_ctrl(s, 1);  // AEC on
    s->set_aec2(s, 1);
    s->set_gain_ctrl(s, 1);      // AGC on
    s->set_hmirror(s, 1);
    s->set_vflip(s, 1);

    Serial.println("Camera OK");
    return true;
}

// ── Beacon-based clock sync ────────────────────────────────────────────────────
// Laptop embeds its Unix timestamp (µs) in the beacon: "LAPTOP:<ip>:<unix_us>"
// We call settimeofday() directly — no SNTP, no port 123, no sudo needed.
static void syncTimeFromBeacon(int64_t laptop_us) {
    struct timeval tv;
    tv.tv_sec  = (time_t)(laptop_us / 1000000LL);
    tv.tv_usec = (suseconds_t)(laptop_us % 1000000LL);
    settimeofday(&tv, NULL);
    udp_log("Clock synced from beacon — %ld.%06ld", (long)tv.tv_sec, (long)tv.tv_usec);
}

// ── Task: capture at target FPS ───────────────────────────────────────────────
void captureTask(void *) {
    TickType_t lastWake   = xTaskGetTickCount();
    int        last_q     = g_jpeg_quality;
    int        last_fps   = g_target_fps;
    int        last_fs    = g_framesize;
    sensor_t  *s          = esp_camera_sensor_get();

    // Apply initial quality (framesize already set in initCamera)
    s->set_quality(s, last_q);

    while (true) {
        if (g_ota_active) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        // Apply settings only when they change
        int q = g_jpeg_quality;
        if (q != last_q) {
            s->set_quality(s, q);
            last_q = q;
            udp_log("Quality → %d", q);
        }

        int fs = g_framesize;
        if (fs != last_fs) {
            s->set_framesize(s, (framesize_t)fs);
            last_fs = fs;
            lastWake = xTaskGetTickCount();   // reset pacing — frame size affects timing
            // Flush stale frames still in the sensor FIFO from the old resolution
            for (int i = 0; i < 3; i++) {
                camera_fb_t *flush = esp_camera_fb_get();
                if (flush) esp_camera_fb_return(flush);
            }
            udp_log("Resolution → framesize %d", fs);
        }

        // Reset pacing baseline when fps changes to avoid stale lastWake
        int fps = g_target_fps;
        if (fps != last_fps) {
            last_fps = fps;
            lastWake = xTaskGetTickCount();
        }
        TickType_t period = pdMS_TO_TICKS(1000 / fps);

        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            FrameEnvelope env = {
                .fb    = fb,
                .ts_us = (int64_t)tv.tv_sec * 1000000LL + tv.tv_usec
            };
            if (xQueueSend(frameQueue, &env, 0) != pdTRUE) {
                esp_camera_fb_return(fb);
                udp_log("WARN: frame queue full — dropped frame");
            }
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

        if (g_ota_active) {
            esp_camera_fb_return(env.fb);   // free buffer, don't transmit
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

            if (g_laptop_ip[0] == '\0') continue;  // not yet discovered
            udp.beginPacket(g_laptop_ip, LAPTOP_PORT);
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
                udp_log("FPS → %d", val);
            }
        } else if (buf[0] == 'r' && buf[1] == ':') {
            int val = atoi(buf + 2);
            if (val >= 0 && val <= 13) {
                g_framesize = val;
            }
        } else {
            // Sensor settings — applied directly via sensor API
            sensor_t *s = esp_camera_sensor_get();
            if (!s) { udp_log("Sensor unavailable"); continue; }

            if      (strncmp(buf, "br:", 3) == 0) { s->set_brightness(s, atoi(buf+3));      udp_log("Brightness → %d", atoi(buf+3)); }
            else if (strncmp(buf, "ct:", 3) == 0) { s->set_contrast(s, atoi(buf+3));        udp_log("Contrast → %d", atoi(buf+3)); }
            else if (strncmp(buf, "sa:", 3) == 0) { s->set_saturation(s, atoi(buf+3));      udp_log("Saturation → %d", atoi(buf+3)); }
            else if (strncmp(buf, "ae:", 3) == 0) {
                int v = atoi(buf+3);
                s->set_exposure_ctrl(s, v);
                s->set_aec2(s, v);  // OV2640 has two AEC stages — both must match
                udp_log("Auto-exposure → %d (aec+aec2)", v);
            }
            else if (strncmp(buf, "ev:", 3) == 0) {
                int v = atoi(buf+3);
                s->set_exposure_ctrl(s, 0);  // ensure auto off before setting manual
                s->set_aec2(s, 0);
                s->set_aec_value(s, v);
                udp_log("Exposure → %d (auto off)", v);
            }
            else if (strncmp(buf, "ag:", 3) == 0) { s->set_gain_ctrl(s, atoi(buf+3));       udp_log("Auto-gain → %d", atoi(buf+3)); }
            else if (strncmp(buf, "gv:", 3) == 0) {
                int v = atoi(buf+3);
                s->set_gain_ctrl(s, 0);  // ensure auto off before setting manual
                s->set_agc_gain(s, v);
                udp_log("Gain → %d (auto off)", v);
            }
            else if (strncmp(buf, "al:", 3) == 0) { s->set_ae_level(s, atoi(buf+3));        udp_log("AE level → %d", atoi(buf+3)); }
            else if (strncmp(buf, "hm:", 3) == 0) { s->set_hmirror(s, atoi(buf+3));         udp_log("H-mirror → %d", atoi(buf+3)); }
            else if (strncmp(buf, "vf:", 3) == 0) { s->set_vflip(s, atoi(buf+3));           udp_log("V-flip → %d", atoi(buf+3)); }
            else if (strncmp(buf, "wm:", 3) == 0) { s->set_wb_mode(s, atoi(buf+3));         udp_log("WB mode → %d", atoi(buf+3)); }
            else if (strncmp(buf, "rst", 3) == 0) { udp_log("Restarting via software reset..."); delay(200); ESP.restart(); }
            else { Serial.printf("Unknown cmd: %s\n", buf); }
        }
    }
}

// ── Task: beacon-based laptop discovery ───────────────────────────────────────
// Broadcasts "CAM:<id>" every 10s so discover.py can find our IP.
// Listens for "LAPTOP:<ip>" from the receiver so we know where to stream.
void discoveryTask(void *) {
    discUdp.begin(DISCOVERY_PORT);

    char announce[16];
    snprintf(announce, sizeof(announce), "CAM:%d", CAM_ID);

    TickType_t lastAnnounce = xTaskGetTickCount() - pdMS_TO_TICKS(10000); // fire immediately

    while (true) {
        // Re-announce every 10s
        if (xTaskGetTickCount() - lastAnnounce >= pdMS_TO_TICKS(10000)) {
            IPAddress bcast = WiFi.broadcastIP();
            discUdp.beginPacket(bcast, DISCOVERY_PORT);
            discUdp.write((const uint8_t *)announce, strlen(announce));
            discUdp.endPacket();
            Serial.printf("Announced: %s  (broadcast %s)\n",
                          announce, bcast.toString().c_str());
            lastAnnounce = xTaskGetTickCount();
        }

        // Listen for laptop beacon: "LAPTOP:<ip>:<unix_us>"
        int len = discUdp.parsePacket();
        if (len > 0) {
            char buf[80] = {};
            discUdp.read(buf, sizeof(buf) - 1);
            if (strncmp(buf, "LAPTOP:", 7) == 0) {
                char *rest   = buf + 7;
                char *colon2 = strchr(rest, ':');

                char new_ip[24] = {};
                int64_t beacon_us = 0;

                if (colon2) {
                    size_t ip_len = colon2 - rest;
                    if (ip_len < sizeof(new_ip))
                        memcpy(new_ip, rest, ip_len);
                    beacon_us = atoll(colon2 + 1);
                } else {
                    strncpy(new_ip, rest, sizeof(new_ip) - 1);
                }

                bool changed = strncmp(g_laptop_ip, new_ip, sizeof(g_laptop_ip)) != 0;
                strncpy(g_laptop_ip,      new_ip, sizeof(g_laptop_ip) - 1);
                strncpy(g_last_laptop_ip, new_ip, sizeof(g_last_laptop_ip) - 1);
                if (changed)
                    udp_log("Laptop discovered: %s", g_laptop_ip);

                // Sync clock from beacon timestamp (no NTP/port-123 needed)
                if (beacon_us > 1000000000000000LL)   // sanity: > year 2001 in µs
                    syncTimeFromBeacon(beacon_us);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// ── Task: handle OTA updates ──────────────────────────────────────────────────
void otaTask(void *) {
    ArduinoOTA.setHostname(OTA_HOSTNAME);
    ArduinoOTA.setPassword(OTA_PASSWORD);

    ArduinoOTA.onStart([]() {
        g_ota_active = true;
        udp_log("OTA started — stream paused for full bandwidth");
    });
    ArduinoOTA.onEnd([]() {
        udp_log("OTA done — rebooting");
        g_ota_active = false;
    });
    ArduinoOTA.onError([](ota_error_t e) {
        g_ota_active = false;
        udp_log("OTA error [%u] — stream resumed", e);
    });

    ArduinoOTA.begin();
    Serial.println("OTA ready on port 3232");

    while (true) {
        ArduinoOTA.handle();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ── LED helpers (GPIO 33, active LOW) ────────────────────────────────────────
// Blink pattern meanings:
//   fast blink (100ms) : connecting to WiFi
//   slow blink (500ms) : WiFi OK, waiting for laptop beacon
//   solid ON           : streaming (laptop discovered)
//   3 rapid flashes    : camera init failed
#define LED_PIN 33

static void ledOn()  { digitalWrite(LED_PIN, LOW);  }
static void ledOff() { digitalWrite(LED_PIN, HIGH); }
static void ledBlink(int times, int ms) {
    for (int i = 0; i < times; i++) {
        ledOn();  delay(ms);
        ledOff(); delay(ms);
    }
}

void ledTask(void *) {
    while (true) {
        if (g_laptop_ip[0] != '\0') {
            // Laptop discovered — solid on
            ledOn();
            vTaskDelay(pdMS_TO_TICKS(500));
        } else if (WiFi.status() == WL_CONNECTED) {
            // WiFi OK, no laptop yet — slow blink
            ledOn();  vTaskDelay(pdMS_TO_TICKS(500));
            ledOff(); vTaskDelay(pdMS_TO_TICKS(500));
        } else {
            // No WiFi — fast blink
            ledOn();  vTaskDelay(pdMS_TO_TICKS(100));
            ledOff(); vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
}

// ── Task: NTP-style round-trip time sync ──────────────────────────────────────
// Protocol (Cristian's algorithm):
//   ESP32 → laptop:  "SYNC_REQ:<cam_id>:<T1_mono_us>"
//   laptop → ESP32:  "SYNC_RESP:<T1_mono_us>:<T2_laptop_us>:<T3_laptop_us>"
//   ESP32 computes:  RTT = T4_mono - T1_mono
//                    corrected_time = T3_laptop + RTT/2
// Both cameras sync to the same laptop clock → inter-cam error ≈ |RTT1-RTT2|/2
void timeSyncTask(void *) {
    tsUdp.begin(TIMESYNC_PORT);

    while (true) {
        if (g_ota_active) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        if (!g_laptop_ip[0]) {
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        // T1: monotonic timestamp (µs since boot) — unaffected by settimeofday
        int64_t t1 = (int64_t)esp_timer_get_time();

        char req[64];
        snprintf(req, sizeof(req), "SYNC_REQ:%d:%lld", CAM_ID, (long long)t1);
        tsUdp.beginPacket(g_laptop_ip, TIMESYNC_PORT);
        tsUdp.write((const uint8_t *)req, strlen(req));
        tsUdp.endPacket();

        // Wait up to 300 ms for response
        uint32_t deadline_ms = millis() + 300;
        bool synced = false;
        while (millis() < deadline_ms && !synced) {
            int len = tsUdp.parsePacket();
            if (len > 0) {
                char buf[128] = {};
                tsUdp.read(buf, sizeof(buf) - 1);
                if (strncmp(buf, "SYNC_RESP:", 10) == 0) {
                    char *p = buf + 10;
                    int64_t t1_echo   = strtoll(p, &p, 10); p++;
                    int64_t t2_laptop = strtoll(p, &p, 10); p++;
                    int64_t t3_laptop = strtoll(p, &p, 10);

                    int64_t t4  = (int64_t)esp_timer_get_time();
                    int64_t rtt = t4 - t1_echo;          // round-trip in µs
                    int64_t corrected_us = t3_laptop + rtt / 2;

                    struct timeval tv;
                    tv.tv_sec  = (time_t)(corrected_us / 1000000LL);
                    tv.tv_usec = (suseconds_t)(corrected_us % 1000000LL);
                    settimeofday(&tv, NULL);

                    udp_log("TimeSync: RTT=%lldus one-way~%lldus", (long long)rtt, (long long)(rtt / 2));
                    synced = true;
                }
            }
            vTaskDelay(pdMS_TO_TICKS(5));
        }
        if (!synced)
            udp_log("TimeSync: no response from laptop");

        vTaskDelay(pdMS_TO_TICKS(10000));   // re-sync every 10 s
    }
}

// ── Task: WiFi watchdog — reconnects if connection drops ──────────────────────
void wifiTask(void *) {
    while (true) {
        if (WiFi.status() != WL_CONNECTED) {
            udp_log("WiFi lost — reconnecting (IP was: %s)",
                    g_laptop_ip[0] ? g_laptop_ip : "none");
            g_laptop_ip[0] = '\0';   // clear so stream pauses until rediscovered
            WiFi.disconnect();
            WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

            int retries = 0;
            while (WiFi.status() != WL_CONNECTED && retries++ < 20) {
                vTaskDelay(pdMS_TO_TICKS(500));
            }

            if (WiFi.status() == WL_CONNECTED) {
                udp_log("WiFi reconnected — IP: %s",
                        WiFi.localIP().toString().c_str());

                // Clock will re-sync automatically once beacon resumes
            } else {
                udp_log("WiFi reconnect failed — will retry");
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5000));  // check every 5s
    }
}

// ── Arduino entry points ──────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n== ESP32-CAM UDP stream ==");

    pinMode(LED_PIN, OUTPUT);
    ledOff();

    if (!initCamera()) {
        Serial.println("FATAL: halting");
        // 3 rapid flashes = camera init failed
        while (true) { ledBlink(3, 100); delay(500); }
    }

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("Connecting to %s", WIFI_SSID);
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print('.');
        ledOn();  delay(100);
        ledOff(); delay(400);
    }
    Serial.printf("\nWi-Fi OK — IP: %s\n", WiFi.localIP().toString().c_str());
    // Clock syncs from laptop beacon — no NTP server needed

    udp.begin(LAPTOP_PORT);

    frameQueue = xQueueCreate(2, sizeof(FrameEnvelope));
    xTaskCreatePinnedToCore(captureTask,   "capture",   4096, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(sendTask,      "send",      8192, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(cmdTask,       "cmd",       4096, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(otaTask,       "ota",       8192, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(discoveryTask, "discovery", 4096, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(timeSyncTask,  "timesync",  4096, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(ledTask,       "led",       2048, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(wifiTask,      "wifi_wd",   4096, NULL, 1, NULL, 1);
}

void loop() {
    vTaskDelay(portMAX_DELAY);
}
