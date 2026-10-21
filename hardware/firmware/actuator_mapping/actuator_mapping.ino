#define SOLENOID_A 2
#define SOLENOID_Y 3
#define SOLENOID_RIGHT 4
#define SOLENOID_LEFT 5

void setup() 
{
  Serial.begin(115200);

  // set pin mode for all pins
  pinMode(SOLENOID_A, OUTPUT);
  pinMode(SOLENOID_Y, OUTPUT);
  pinMode(SOLENOID_RIGHT, OUTPUT);
  pinMode(SOLENOID_LEFT, OUTPUT);

  // write all low
  digitalWrite(SOLENOID_A, LOW);
  digitalWrite(SOLENOID_Y, LOW);
  digitalWrite(SOLENOID_RIGHT, LOW);
  digitalWrite(SOLENOID_LEFT, LOW);
}

void loop() 
{
  if (Serial.available() > 0) {
    char command = Serial.read();
    pulse(command);
  }
}

void pulse(char command)
{
  switch (command)
  {
    case '1':
      Serial.println("Pulsing A...");
      digitalWrite(SOLENOID_A, HIGH);
      delay(100);
      digitalWrite(SOLENOID_A, LOW);
      break;
    case '2':
      Serial.println("Pulsing Y...");
      digitalWrite(SOLENOID_Y, HIGH);
      delay(100);
      digitalWrite(SOLENOID_Y, LOW);
      break;
    case '3':
      Serial.println("Pulsing RIGHT (D-PAD)...");
      digitalWrite(SOLENOID_RIGHT, HIGH);
      delay(100);
      digitalWrite(SOLENOID_RIGHT, LOW);
      break;
    case '4':
      Serial.println("Pulsing LEFT (D-PAD)...");
      digitalWrite(SOLENOID_LEFT, HIGH);
      delay(100);
      digitalWrite(SOLENOID_LEFT, LOW);
      break;
  }
}