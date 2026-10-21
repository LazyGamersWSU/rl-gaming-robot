# Electronics

The electronics system controls and powers the solenoid actuators used to physically operate the game controller.

## Subsystems

- [PCB](pcb/) — Solenoid driver PCB design and fabrication files with revision history
- [Power](power/power_budget.md) — Battery and system power requirements
- [Wiring](wiring/) — Electrical wiring and connections (visual)

## Signal Path

Arduino → MOSFET driver → Solenoids

## Power Path

Battery → BMS → Power switch → Fuse  → Solenoid driver PCB → Solenoids