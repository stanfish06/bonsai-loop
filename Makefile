.PHONY: bonsai cellstates clean format-project lint

# bonsai_port checkout and build folder for the C++ Bonsai executable
BONSAI_PORT_DIR ?= packages/pybonsai/third_party/bonsai_port
BONSAI_BUILD_DIR ?= build/bonsai

# Build the C++ Bonsai executable at $(BONSAI_BUILD_DIR)/bonsai
bonsai:
	cmake -S $(BONSAI_PORT_DIR) -B $(BONSAI_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
	cmake --build $(BONSAI_BUILD_DIR) --target libbonsai --parallel
	cp $(BONSAI_BUILD_DIR)/libbonsai $(BONSAI_BUILD_DIR)/bonsai

# supercellstates checkout and build folder for the cellstates executable
CELLSTATES_DIR ?= third_party/supercellstates
CELLSTATES_BUILD_DIR ?= build/cellstates

# Build the cellstates executable and coarsify_cellstates, which merges cellstates down to a
# target number, in $(CELLSTATES_BUILD_DIR)
cellstates:
	mkdir -p $(CELLSTATES_BUILD_DIR)
	$(CXX) -std=c++20 -O3 -ffast-math -fopenmp $(CELLSTATES_DIR)/cluster.cpp -o $(CELLSTATES_BUILD_DIR)/cellstates -lz
	$(CXX) -std=c++20 -O3 -ffast-math -fopenmp $(CELLSTATES_DIR)/coarsify_cellstates.cpp -o $(CELLSTATES_BUILD_DIR)/coarsify_cellstates -lz

format-project:
	uvx pyproject-fmt pyproject.toml || true
	uvx docformatter --in-place --pre-summary-newline --recursive --wrap-summaries 88 --wrap-descriptions 88 src/bonsai_loop/ || true
	uv run ruff format || true
	uv run ruff check --fix || true


lint:
	uv run pyright || true
	uv run ruff check src/ || true
	uv run --with mypy mypy --ignore-missing-imports src/ || true
	uv run pyrefly check src/ || true

clean:
	uv cache clean
	uv cache prune
	rm -rf .venv
	rm -rf dist

