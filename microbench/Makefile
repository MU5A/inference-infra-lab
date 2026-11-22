# Day 2 microbenchmarks. `make` builds into ./build, `make test` runs each program's self-test.
CC ?= cc
BUILD ?= build
CFLAGS ?= -O3 -std=gnu11 -Wall -Wextra -pthread
LDLIBS = -lm

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  CFLAGS += -mcpu=native -DACCELERATE_NEW_LAPACK
  MATMUL_LIBS = -framework Accelerate
else
  CFLAGS += -march=native
  MATMUL_LIBS =
endif

PROGS = membw chase matmul mmap_touch copy_nocache
TARGETS = $(addprefix $(BUILD)/,$(PROGS))

all: $(TARGETS)

$(BUILD)/%: %.c common.h
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS) $(if $(filter matmul,$*),$(MATMUL_LIBS),)

test: all
	$(BUILD)/membw --selftest
	$(BUILD)/chase --selftest
	$(BUILD)/matmul --selftest

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
