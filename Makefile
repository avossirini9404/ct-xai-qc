# Reproduce everything: make setup && make all
CONFIG ?= configs/default.yaml
PY     ?= python

.PHONY: setup data train evaluate experiment report test lint smoke all clean

setup:            ## install the package and its dependencies
	$(PY) -m pip install -e ".[data,dev]"

data:             ## download the MedMNIST split declared in $(CONFIG)
	$(PY) -c "from ctxaiqc.utils import load_config; from ctxaiqc.data import load_dataset; \
	cfg=load_config('$(CONFIG)'); print(load_dataset(**cfg['dataset'], seed=cfg['seed']).summary())"

train:            ## train the classifier
	$(PY) -m ctxaiqc.train --config $(CONFIG)

evaluate:         ## test-set performance and calibration of the checkpoint
	$(PY) -m ctxaiqc.evaluate --config $(CONFIG)

experiment:       ## run the quality-control protocol, writing results/tables/*.csv
	$(PY) -m ctxaiqc.run_experiment --config $(CONFIG)

report:           ## build the figures and results/tables/summary.md
	$(PY) -m ctxaiqc.report --config $(CONFIG)

test:             ## offline unit and smoke tests
	$(PY) -m pytest -q

lint:
	ruff check src tests

smoke:            ## the whole pipeline on synthetic data, no download
	$(MAKE) train CONFIG=configs/smoke.yaml
	$(MAKE) experiment CONFIG=configs/smoke.yaml
	$(MAKE) report CONFIG=configs/smoke.yaml

all: train experiment report

clean:
	rm -rf results/tables results/figures results/*.pt results/smoke
