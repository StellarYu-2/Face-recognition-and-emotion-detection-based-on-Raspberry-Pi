#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "your_wifi";
const char* WIFI_PASSWORD = "your_password";

// Use your public Platform Server base URL. Do not include a trailing slash.
const char* PLATFORM_BASE_URL = "https://api.asdun.example.com";

const char* DEVICE_ID = "esp32-01";
const char* DEVICE_TOKEN = "replace-with-esp32-token";

unsigned long lastTelemetryMs = 0;
unsigned long lastCommandPollMs = 0;

const unsigned long TELEMETRY_INTERVAL_MS = 5000;
const unsigned long COMMAND_POLL_INTERVAL_MS = 3000;

String currentMode = "normal";

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

void addDeviceHeaders(HTTPClient& http) {
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-ASDUN-Device-Id", DEVICE_ID);
  http.addHeader("X-ASDUN-Device-Token", DEVICE_TOKEN);
}

bool postTelemetry(float temperature, float humidity) {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  HTTPClient http;
  String url = String(PLATFORM_BASE_URL) + "/api/telemetry";
  http.begin(url);
  addDeviceHeaders(http);

  StaticJsonDocument<384> doc;
  doc["device_id"] = DEVICE_ID;
  doc["role"] = "esp32";
  doc["display_name"] = DEVICE_ID;
  doc["online"] = true;
  doc["telemetry"]["temperature"] = temperature;
  doc["telemetry"]["humidity"] = humidity;
  doc["telemetry"]["rssi"] = WiFi.RSSI();
  doc["telemetry"]["uptime_ms"] = millis();
  doc["telemetry"]["mode"] = currentMode;

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  http.end();
  return code >= 200 && code < 300;
}

bool postCommandResult(const String& commandId, bool ok, const String& message) {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  HTTPClient http;
  String url = String(PLATFORM_BASE_URL) + "/api/commands/" + commandId + "/result";
  http.begin(url);
  addDeviceHeaders(http);

  StaticJsonDocument<384> doc;
  doc["device_id"] = DEVICE_ID;
  doc["ok"] = ok;
  doc["message"] = message;
  doc["result"]["mode"] = currentMode;
  doc["result"]["handled_by"] = DEVICE_ID;
  doc["result"]["uptime_ms"] = millis();

  String body;
  serializeJson(doc, body);

  int code = http.POST(body);
  http.end();
  return code >= 200 && code < 300;
}

void handleCommand(JsonObject command) {
  const char* commandId = command["command_id"] | "";
  const char* name = command["command"] | "";
  JsonObject payload = command["payload"].as<JsonObject>();

  if (strlen(commandId) == 0 || strlen(name) == 0) {
    return;
  }

  if (strcmp(name, "ping") == 0) {
    postCommandResult(commandId, true, "pong");
    return;
  }

  if (strcmp(name, "set_mode") == 0) {
    const char* mode = payload["mode"] | "normal";
    currentMode = String(mode);
    postCommandResult(commandId, true, "mode updated");
    return;
  }

  postCommandResult(commandId, false, "unsupported command");
}

bool pollCommands() {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  HTTPClient http;
  String url = String(PLATFORM_BASE_URL) + "/api/commands/pending?device_id=" + DEVICE_ID + "&limit=5";
  http.begin(url);
  addDeviceHeaders(http);

  int code = http.GET();
  if (code < 200 || code >= 300) {
    http.end();
    return false;
  }

  String response = http.getString();
  http.end();

  StaticJsonDocument<2048> doc;
  DeserializationError error = deserializeJson(doc, response);
  if (error) {
    return false;
  }

  JsonArray commands = doc["commands"].as<JsonArray>();
  for (JsonObject command : commands) {
    handleCommand(command);
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  connectWifi();
  Serial.println("ESP32 platform client started");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWifi();
  }

  unsigned long now = millis();

  if (now - lastTelemetryMs >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryMs = now;

    // Replace these placeholders with real sensor reads.
    float temperature = 28.4;
    float humidity = 61.0;

    bool ok = postTelemetry(temperature, humidity);
    Serial.println(ok ? "telemetry ok" : "telemetry failed");
  }

  if (now - lastCommandPollMs >= COMMAND_POLL_INTERVAL_MS) {
    lastCommandPollMs = now;

    bool ok = pollCommands();
    Serial.println(ok ? "command poll ok" : "command poll failed");
  }
}
