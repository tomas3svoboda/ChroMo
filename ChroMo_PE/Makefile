CMAKE_BUILD_DIR := build
CMAKE_GENERATOR :=
BUILD_TYPE := Release
BINARY_NAME := edm_nonlinear_solver

.PHONY: all configure build copy clean rebuild

all: configure build copy

configure:
	mkdir -p $(CMAKE_BUILD_DIR)
	cd $(CMAKE_BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_GENERATOR) ..

build: configure
	$(MAKE) -C $(CMAKE_BUILD_DIR)

# Copies the built binary from build/ to current directory
copy:
	@cp -f $(CMAKE_BUILD_DIR)/$(BINARY_NAME)*.so ./$(BINARY_NAME).so || { echo "Binary not found: $(CMAKE_BUILD_DIR)/$(BINARY_NAME).so"; exit 1; }

rebuild: clean all

clean:
	-rm -rf $(CMAKE_BUILD_DIR) $(BINARY_NAME)
