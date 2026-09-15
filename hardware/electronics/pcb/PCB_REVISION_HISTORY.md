# PCB Revision History

This document tracks revisions to the RL Gaming Robot solenoid driver PCB and summarizes the changes made between each hardware revision.

## Revision History

| Revision | Status                       | Description                                                                                                                                                                                                                                                                                                                           |
| -------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v1.0** | Initial Implementation       | Initial implementation of the solenoid driver PCB. Established the basic circuit design for controlling the solenoids from the Arduino, including MOSFET switching, flyback protection, power distribution, and LED indicator circuitry.                                                                                                  |
| **v2.0** | WSU Electronics Lab Revision | Modified the PCB design to support fabrication using the single-sided PCB manufacturing process available through the WSU Electronics Lab. The ground plane was removed, wire traces were moved to the bottom copper layer, and mounting/component holes were resized to match the available router bits used for PCB fabrication.    |
| **v3.0** | Circuit Correction           | Corrected an error in the MOSFET control circuit identified during design review. The pull-down resistors had originally been connected to the MOSFET drain pins rather than the gate pins. The resistors were moved to the gate pins so they properly hold the MOSFETs in the off state when the Arduino control signal is inactive. |

## Revision Guidelines

* Increment the PCB revision when the schematic or physical PCB design changes.
* Record the purpose and significant changes associated with each revision.
* PCB revisions refer to the physical/electrical design and are independent of software or repository versions.
