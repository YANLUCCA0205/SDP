
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_heap_caps.h"

// ============================================================
// ESP32-CAM AI-THINKER | TCC SDP
// DIAGNOSTICO v2.0
// ============================================================

// WIFI - PREENCHA A SENHA LOCALMENTE
const char* ssid = "COLOQUE_SEU_SSID_AQUI";
const char* password = "COLOQUE_SUA_SENHA_AQUI";

// API PYTHON - ENDERECO ATUAL NA REDE LOCAL
const char* apiHost = "COLOQUE_O_IP_DO_SERVIDOR_AQUI";
const uint16_t apiPort = 5000;
const char* apiPath = "/captures?camera_id=cam-01";

// CONFIGURACOES
const unsigned long INTERVALO_MS = 60000;
const bool USAR_FLASH = false;
const bool DESCARTAR_PRIMEIRO_FRAME = true;

// Se a captura continuar falhando, testar FRAMESIZE_QVGA
#define RESOLUCAO_TESTE FRAMESIZE_VGA

// 0 = maior qualidade / 63 = menor qualidade
const int QUALIDADE_JPEG = 20;
const int MAX_TENTATIVAS = 2;

// Iniciamos com 20 MHz porque 10 MHz nao resolveu
const int CLOCK_CAMERA = 20000000;

#define FLASH_LED_PIN 4

// PINOS AI-THINKER
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5

#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// CONTADORES
unsigned long sequencia = 0;
unsigned long capturasOK = 0;
unsigned long falhasCaptura = 0;
unsigned long enviosOK = 0;
unsigned long falhasEnvio = 0;
unsigned long ultimoCiclo = 0;

bool psramOK = false;

// ============================================================
// RELATORIOS
// ============================================================

void relatorioMemoria(const char* etapa) {
  Serial.printf(
    "[MEM] etapa=%s heap=%u min_heap=%u psram_livre=%u\n",
    etapa,
    (unsigned)ESP.getFreeHeap(),
    (unsigned)ESP.getMinFreeHeap(),
    (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM)
  );
}

void relatorioSistema() {
  Serial.println("\n========== SISTEMA ==========");
  Serial.printf("[SYS] chip=%s\n", ESP.getChipModel());
  Serial.printf("[SYS] cpu_mhz=%u\n", ESP.getCpuFreqMHz());
  Serial.printf("[SYS] flash_bytes=%u\n",
                (unsigned)ESP.getFlashChipSize());
  Serial.printf("[SYS] psram_detectada=%s\n",
                psramOK ? "SIM" : "NAO");
  Serial.printf("[SYS] psram_total=%u\n",
                (unsigned)ESP.getPsramSize());
  Serial.printf("[SYS] psram_livre=%u\n",
                (unsigned)ESP.getFreePsram());
  relatorioMemoria("INICIO");
  Serial.println("=============================\n");
}

void relatorioResumo() {
  Serial.println("\n========== RESUMO ==========");
  Serial.printf(
    "[STAT] ciclos=%lu captura_ok=%lu captura_falha=%lu "
    "envio_ok=%lu envio_falha=%lu\n",
    sequencia,
    capturasOK,
    falhasCaptura,
    enviosOK,
    falhasEnvio
  );
  relatorioMemoria("FIM_CICLO");
  Serial.println("============================\n");
}

// ============================================================
// WIFI
// ============================================================

bool conectarWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }

  Serial.println("[WIFI] conectando...");

  WiFi.begin(ssid, password);

  unsigned long inicio = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - inicio < 15000) {
    delay(250);
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.printf("[WIFI] FALHA status=%d\n",
                  (int)WiFi.status());
    return false;
  }

  Serial.print("[WIFI] conectado ip=");
  Serial.println(WiFi.localIP());

  return true;
}

void relatorioWiFi() {
  Serial.printf(
    "[WIFI] status=%d rssi=%d dBm\n",
    (int)WiFi.status(),
    WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0
  );
}

// ============================================================
// CAMERA
// ============================================================

bool iniciarCamera() {
  camera_config_t config = {};

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

  config.xclk_freq_hz = CLOCK_CAMERA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = RESOLUCAO_TESTE;
  config.jpeg_quality = QUALIDADE_JPEG;

  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

  config.fb_location = psramOK
    ? CAMERA_FB_IN_PSRAM
    : CAMERA_FB_IN_DRAM;

  Serial.println("[CAM] inicializando...");
  Serial.printf("[CAM] xclk=%d\n", CLOCK_CAMERA);
  Serial.printf("[CAM] jpeg_quality=%d\n", QUALIDADE_JPEG);
  Serial.printf("[CAM] fb_count=%d\n", config.fb_count);
  Serial.printf("[CAM] memoria=%s\n",
                psramOK ? "PSRAM" : "DRAM");

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("[CAM] INIT_FALHOU erro=0x%x\n", err);
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();

  if (sensor != NULL) {
    Serial.printf("[CAM] sensor_pid=0x%x\n",
                  sensor->id.PID);

    Serial.printf(
      "[CAM] exposicao_auto=%d ganho_auto=%d\n",
      sensor->status.aec,
      sensor->status.agc
    );
  }

  Serial.println("[CAM] inicializada=SIM");
  relatorioMemoria("CAMERA_INICIALIZADA");

  return true;
}

// ============================================================
// VALIDACAO JPEG
// ============================================================

bool jpegValido(camera_fb_t* fb) {
  if (fb == NULL || fb->buf == NULL || fb->len < 4) {
    return false;
  }

  bool inicioOK =
    fb->buf[0] == 0xFF &&
    fb->buf[1] == 0xD8;

  bool finalOK =
    fb->buf[fb->len - 2] == 0xFF &&
    fb->buf[fb->len - 1] == 0xD9;

  Serial.printf(
    "[JPEG] inicio=%s final=%s bytes=%u\n",
    inicioOK ? "OK" : "INVALIDO",
    finalOK ? "OK" : "INVALIDO",
    (unsigned)fb->len
  );

  return inicioOK && finalOK;
}

// ============================================================
// CAPTURA
// ============================================================

camera_fb_t* capturarImagem() {
  for (int tentativa = 1;
       tentativa <= MAX_TENTATIVAS;
       tentativa++) {

    Serial.printf(
      "[CAP] tentativa=%d/%d\n",
      tentativa,
      MAX_TENTATIVAS
    );

    unsigned long inicio = millis();

    if (USAR_FLASH) {
      digitalWrite(FLASH_LED_PIN, HIGH);
    }

    delay(400);

    // Mantem o comportamento que resolveu a foto atrasada.
    if (DESCARTAR_PRIMEIRO_FRAME) {
      Serial.println("[CAP] descartando_frame_inicial");

      camera_fb_t* anterior = esp_camera_fb_get();

      if (anterior != NULL) {
        Serial.printf(
          "[CAP] frame_descartado_bytes=%u\n",
          (unsigned)anterior->len
        );
        esp_camera_fb_return(anterior);
      } else {
        Serial.println("[CAP] descarte_falhou");
      }
    }

    Serial.println("[CAP] solicitando_frame_atual");

    camera_fb_t* fb = esp_camera_fb_get();

    digitalWrite(FLASH_LED_PIN, LOW);

    unsigned long duracao = millis() - inicio;

    if (fb == NULL) {
      Serial.printf(
        "[CAP] ERRO frame_nulo tempo_ms=%lu\n",
        duracao
      );

      delay(500);
      continue;
    }

    Serial.printf(
      "[CAP] recebido bytes=%u largura=%u "
      "altura=%u tempo_ms=%lu\n",
      (unsigned)fb->len,
      (unsigned)fb->width,
      (unsigned)fb->height,
      duracao
    );

    if (!jpegValido(fb)) {
      Serial.println("[CAP] ERRO jpeg_invalido");
      esp_camera_fb_return(fb);
      delay(500);
      continue;
    }

    capturasOK++;
    Serial.println("[CAP] RESULTADO=OK");

    return fb;
  }

  falhasCaptura++;
  Serial.println("[CAP] RESULTADO=FALHA");

  return NULL;
}

// ============================================================
// ESCRITA HTTP COM CONTAGEM REAL DE BYTES
// ============================================================

bool escreverTudo(
  WiFiClient& client,
  const uint8_t* dados,
  size_t tamanho,
  const char* etapa
) {
  size_t enviados = 0;
  unsigned long ultimoProgresso = millis();

  while (enviados < tamanho) {
    if (!client.connected()) {
      Serial.printf(
        "[HTTP] %s conexao_perdida enviados=%u/%u\n",
        etapa,
        (unsigned)enviados,
        (unsigned)tamanho
      );
      return false;
    }

    size_t bloco = tamanho - enviados;
    if (bloco > 1024) bloco = 1024;

    size_t escritos = client.write(
      dados + enviados,
      bloco
    );

    if (escritos > 0) {
      enviados += escritos;
      ultimoProgresso = millis();
    } else {
      if (millis() - ultimoProgresso > 10000) {
        Serial.printf(
          "[HTTP] %s TIMEOUT enviados=%u/%u\n",
          etapa,
          (unsigned)enviados,
          (unsigned)tamanho
        );
        return false;
      }

      delay(10);
    }
  }

  Serial.printf(
    "[HTTP] %s bytes=%u/%u\n",
    etapa,
    (unsigned)enviados,
    (unsigned)tamanho
  );

  return true;
}

// ============================================================
// ENVIO PARA API
// ============================================================

bool enviarImagem(camera_fb_t* fb) {
  if (!conectarWiFi()) {
    Serial.println("[HTTP] erro=SEM_WIFI");
    return false;
  }

  relatorioWiFi();

  WiFiClient client;
  client.setTimeout(3000);

  Serial.printf(
    "[HTTP] destino=%s:%u%s\n",
    apiHost,
    apiPort,
    apiPath
  );

  unsigned long inicioConexao = millis();

  if (!client.connect(apiHost, apiPort, 5000)) {
    Serial.println("[HTTP] erro=CONEXAO_RECUSADA_OU_TIMEOUT");
    return false;
  }

  Serial.printf(
    "[HTTP] conectado tempo_ms=%lu\n",
    millis() - inicioConexao
  );

  const String boundary = "----ESP32CAMBoundary";

  String head =
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; "
    "name=\"file\"; filename=\"esp32cam.jpg\"\r\n"
    "Content-Type: image/jpeg\r\n\r\n";

  String tail = "\r\n--" + boundary + "--\r\n";

  size_t tamanhoTotal =
    head.length() + fb->len + tail.length();

  String headers =
    "POST " + String(apiPath) + " HTTP/1.1\r\n"
    "Host: " + String(apiHost) + "\r\n"
    "Content-Type: multipart/form-data; boundary=" +
    boundary + "\r\n"
    "Content-Length: " + String(tamanhoTotal) + "\r\n"
    "Connection: close\r\n\r\n";

  Serial.printf(
    "[HTTP] jpeg_bytes=%u multipart_total=%u\n",
    (unsigned)fb->len,
    (unsigned)tamanhoTotal
  );

  unsigned long inicioEnvio = millis();

  bool ok = escreverTudo(
    client,
    (const uint8_t*)headers.c_str(),
    headers.length(),
    "HEADERS"
  );

  if (ok) {
    ok = escreverTudo(
      client,
      (const uint8_t*)head.c_str(),
      head.length(),
      "MULTIPART_HEAD"
    );
  }

  if (ok) {
    ok = escreverTudo(
      client,
      fb->buf,
      fb->len,
      "JPEG"
    );
  }

  if (ok) {
    ok = escreverTudo(
      client,
      (const uint8_t*)tail.c_str(),
      tail.length(),
      "MULTIPART_TAIL"
    );
  }

  if (!ok) {
    Serial.println("[HTTP] RESULTADO=FALHA_ENVIO");
    client.stop();
    return false;
  }

  Serial.printf(
    "[HTTP] dados_escritos tempo_ms=%lu\n",
    millis() - inicioEnvio
  );

  // Aguarda a linha de status HTTP, com timeout.
  unsigned long inicioResposta = millis();

  while (!client.available() &&
         client.connected() &&
         millis() - inicioResposta < 10000) {
    delay(10);
  }

  if (!client.available()) {
    Serial.println("[HTTP] erro=SEM_RESPOSTA");
    client.stop();
    return false;
  }

  String statusLine = client.readStringUntil('\n');
  statusLine.trim();

  Serial.print("[HTTP] resposta=");
  Serial.println(statusLine);

  int statusCode = 0;

  int primeiroEspaco = statusLine.indexOf(' ');

  if (primeiroEspaco >= 0) {
    statusCode = statusLine.substring(
      primeiroEspaco + 1,
      primeiroEspaco + 4
    ).toInt();
  }

  Serial.printf("[HTTP] status_code=%d\n", statusCode);

  client.stop();

  if (statusCode >= 200 && statusCode < 300) {
    Serial.println("[HTTP] RESULTADO=SUCESSO");
    return true;
  }

  Serial.println("[HTTP] RESULTADO=ERRO_API");
  return false;
}

// ============================================================
// CICLO PRINCIPAL
// ============================================================

void executarCiclo() {
  sequencia++;

  unsigned long inicio = millis();

  Serial.println("\n================================");
  Serial.printf("[CICLO] numero=%lu\n", sequencia);
  Serial.printf("[CICLO] uptime_ms=%lu\n", millis());
  Serial.printf(
    "[CICLO] flash=%s\n",
    USAR_FLASH ? "ON" : "OFF"
  );
  Serial.println("================================");

  relatorioMemoria("ANTES_CAPTURA");

  camera_fb_t* fb = capturarImagem();

  if (fb == NULL) {
    Serial.println("[CICLO] envio_ignorado=CAPTURA_FALHOU");
    relatorioResumo();
    return;
  }

  relatorioMemoria("APOS_CAPTURA");

  bool enviado = enviarImagem(fb);

  // O buffer e devolvido mesmo quando o envio falha.
  esp_camera_fb_return(fb);
  fb = NULL;

  if (enviado) {
    enviosOK++;
  } else {
    falhasEnvio++;
  }

  Serial.printf(
    "[CICLO] duracao_ms=%lu\n",
    millis() - inicio
  );

  relatorioResumo();
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("\n================================");
  Serial.println("ESP32-CAM | SDP TCC");
  Serial.println("DIAGNOSTICO v2.0");
  Serial.println("================================");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  psramOK = psramFound();

  relatorioSistema();

  if (!psramOK) {
    Serial.println(
      "[ALERTA] PSRAM indisponivel. "
      "Verifique a configuracao da placa."
    );
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  conectarWiFi();

  if (!iniciarCamera()) {
    Serial.println(
      "[FATAL] Camera nao inicializada. "
      "Verifique hardware e configuracoes."
    );

    while (true) {
      delay(1000);
    }
  }

  // Primeira captura imediata.
  ultimoCiclo = millis() - INTERVALO_MS;

  Serial.println("[SYS] setup_concluido");
}

// ============================================================
// LOOP
// ============================================================

void loop() {
  if (millis() - ultimoCiclo >= INTERVALO_MS) {
    executarCiclo();

    // Intervalo entre o fim de um ciclo e o proximo.
    ultimoCiclo = millis();
  }

  delay(50);
}
