#!/bin/bash

make clean
make
./detect_id_card --debug ./id_samples/ontario_drivers_license_front.jpg
