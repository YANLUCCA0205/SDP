#include <WiFi.h>
#include "esp_camera.h"

// ============================================================
// CONFIGURAÇÕES DO WIFI
// ============================================================

const char* ssid = "ADM";
const char* password = "adm@2025@";

// ============================================================
// CONFIGURAÇÕES DA API PYTHON
// ============================================================

// IP do computador onde o Python estará rodando
const char* apiHost = "192.168.1.128";

// Porta da API
const int apiPort = 5000;

// Endpoint
const char* apiPath = "/captur  -01";

// ============================================================
// CONFIGURAÇÕES DE CAPTURA
// ============================================================

// 30 segundos
const unsigned long timerDelay = 60000;

unsigned long lastTime = 0;

// Flash
bool useFlash = true;

#define FLASH_LED_PIN 4

// ============================================================
// PINOS DA ESP32-CAM AI-THINKER
// ============================================================

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
#define Y2_GPIO_NUM       5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ============================================================
// CONFIGURAÇÃO DA CÂMERA
// ============================================================

void configCamera() {

  camera_config_t config;

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  // JPEG
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {

    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;

  } else {

    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {

    Serial.printf(
      "Falha na inicializacao da camera: 0x%x\n",
      err
    );

    delay(2000);

    ESP.restart();
  }

  Serial.println("Camera inicializada com sucesso.");
}

// ============================================================
// GARANTE WIFI
// ============================================================

void connectWiFi() {

  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("Conectando ao WiFi...");
  Serial.print("Status WiFi: ");
  Serial.println(WiFi.status());

  WiFi.disconnect();
  WiFi.begin(ssid, password);

  int tentativas = 0;

  while (WiFi.status() != WL_CONNECTED && tentativas < 30) {

    delay(500);

    Serial.print(".");

    tentativas++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println("WiFi conectado.");
    Serial.print("IP da ESP32-CAM: ");
    Serial.println(WiFi.localIP());

  } else {

    Serial.println("Nao foi possivel conectar ao WiFi.");
  }
}

// ============================================================
// ENVIO DA FOTO PARA PYTHON
// ============================================================

void sendPhotoToPython() {

  connectWiFi();

  if (WiFi.status() != WL_CONNECTED) {

    Serial.println("WiFi indisponivel.");
    return;
  }

  // ----------------------------------------------------------
  // FLASH
  // ----------------------------------------------------------

  if (useFlash) {

    digitalWrite(FLASH_LED_PIN, HIGH);

    // Pequeno tempo para estabilizar iluminação
    delay(100);
  }

  // ----------------------------------------------------------
  // CAPTURA
  // ----------------------------------------------------------

  camera_fb_t* fb = esp_camera_fb_get();

  // Desliga flash imediatamente
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!fb) {

    Serial.println("Falha ao capturar imagem.");

    return;
  }

  Serial.println("--------------------------------");
  Serial.println("Imagem capturada.");
  Serial.print("Tamanho JPEG: ");
  Serial.print(fb->len);
  Serial.println(" bytes");

  // ----------------------------------------------------------
  // CONEXÃO COM PYTHON
  // ----------------------------------------------------------

  WiFiClient client;

  Serial.print("Conectando na API: ");
  Serial.print(apiHost);
  Serial.print(":");
  Serial.println(apiPort);

  if (!client.connect(apiHost, apiPort)) {

    Serial.println("Falha ao conectar na API.");

    esp_camera_fb_return(fb);

    return;
  }

  Serial.println("Conectado na API.");

  // ----------------------------------------------------------
  // MULTIPART
  // ----------------------------------------------------------

  String boundary = "----ESP32CAMBoundary";

  String head =
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"file\"; filename=\"esp32cam.jpg\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";

  String tail =
      "\r\n--" + boundary + "--\r\n";

  uint32_t totalLength =
      head.length() +
      fb->len +
      tail.length();

  // ----------------------------------------------------------
  // HTTP POST
  // ----------------------------------------------------------

  client.println(
      "POST " + String(apiPath) + " HTTP/1.1"
  );

  client.println(
      "Host: " + String(apiHost)
  );

  client.println(
      "Content-Type: multipart/form-data; boundary=" + boundary
  );

  client.println(
      "Content-Length: " + String(totalLength)
  );

  client.println(
      "Connection: close"
  );

  client.println();

  // ----------------------------------------------------------
  // ENVIA HEADER MULTIPART
  // ----------------------------------------------------------

  client.print(head);

  // ----------------------------------------------------------
  // ENVIA JPEG EM BLOCOS
  // ----------------------------------------------------------

  uint8_t* buffer = fb->buf;

  size_t remaining = fb->len;

  const size_t chunkSize = 1024;

  while (remaining > 0) {

    size_t chunk = remaining;

    if (chunk > chunkSize) {
      chunk = chunkSize;
    }

    client.write(buffer, chunk);

    buffer += chunk;
    remaining -= chunk;
  }

  // ----------------------------------------------------------
  // FINAL MULTIPART
  // ----------------------------------------------------------

  client.print(tail);

  Serial.println("Imagem enviada. Aguardando resposta...");

  // ----------------------------------------------------------
  // LÊ RESPOSTA DA API
  // ----------------------------------------------------------

  unsigned long timeout = millis();

  while (client.connected() &&
         millis() - timeout < 10000) {

    while (client.available()) {

      String line = client.readStringUntil('\n');

      Serial.println(line);

      timeout = millis();
    }
  }

  client.stop();

  // ----------------------------------------------------------
  // LIBERA FRAME BUFFER
  // ----------------------------------------------------------

  esp_camera_fb_return(fb);

  Serial.println("Envio finalizado.");
  Serial.println("--------------------------------");
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println("ESP32-CAM + API Python");
  Serial.println("================================");

  // Flash
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // WiFi
  WiFi.mode(WIFI_STA);

  connectWiFi();

  // Camera
  configCamera();

  // Primeira captura logo após iniciar
  lastTime = millis() - timerDelay;
}

// ============================================================
// LOOP
// ============================================================

void loop() {

  if (millis() - lastTime >= timerDelay) {

    sendPhotoToPython();

    lastTime = millis();
  }
}