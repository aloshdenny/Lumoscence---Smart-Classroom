#include <WiFi.h>
#include <FirebaseESP32.h>

const char* ssid = "asna";
const char* password = "12345678";

#define FIREBASE_HOST "https://class-b2375-default-rtdb.asia-southeast1.firebasedatabase.app/"
#define FIREBASE_AUTH "AIzaSyCiSJH4YZ2MJ_2dQ7aWqJIaZlv2VZ3P41E"

FirebaseData firebaseData;

int motorPins[] = {15, 4, 18, 19, 22, 23};
int ledPins[] = {13,12,14,27,26,25};

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("WiFi connected");

  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
  Firebase.reconnectWiFi(true);

  for (int i = 0; i < 6; i++) {
    pinMode(motorPins[i], OUTPUT);
    pinMode(ledPins[i], OUTPUT);
  }
}

void loop() {
  if (Firebase.getJSON(firebaseData, "/detections")) {
    Serial.println("Fetched data:");
    FirebaseJson& json = firebaseData.jsonObject();
    FirebaseJsonArray countsArray;
    FirebaseJsonData jsonData;
    String jsonString;
    json.toString(jsonString);
    Serial.println(jsonString);

    if (json.get(jsonData, "counts") && jsonData.typeNum == FirebaseJson::JSON_ARRAY) {
      jsonData.getArray(countsArray);
      for (size_t i = 0; i < countsArray.size(); i++) {
        if (countsArray.get(jsonData, i)) {
          Serial.print("Section ");
          Serial.print(i);
          Serial.print(": ");
          Serial.println(jsonData.intValue);
          if (jsonData.intValue > 0 && i < 6) {
            digitalWrite(motorPins[i], HIGH);
            digitalWrite(ledPins[i], HIGH);
          } else {
            digitalWrite(motorPins[i], LOW);
            digitalWrite(ledPins[i], LOW);
          }
        }
      }
    } else {
      Serial.println("Failed to get 'counts' array.");
    }
  } else {
    Serial.println("Failed to fetch data");
    Serial.println("Reason: " + firebaseData.errorReason());
  }

  delay(5000);
}