#!/bin/bash

CC='gcc'
CFLAGS='-O0 --fno-inline-functions -g'

SRC_DIR='/home/caoy/tf/data/transdata/src'
OBJ_DIR='/home/caoy/tf/data/transdata/bin'

SRCS=$(ls ${SRC_DIR})

for src in ${SRCS}
do
    $(CC) $(CFLAGS) $(SRC_DIR)/$(src) -o $(OBJ_DIR)/$(src%?)o
done

